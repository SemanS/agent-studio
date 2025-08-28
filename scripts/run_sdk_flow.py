#!/usr/bin/env python3
import os
import sys
import time
import json
import base64
import urllib.request
import urllib.error
from typing import Dict, Any, Optional

import yaml

try:
    from databricks.sdk import WorkspaceClient
except Exception as e:
    print(f"ERROR: databricks-sdk not installed or failed to import: {e}", file=sys.stderr)
    sys.exit(2)


def parse_env(path: str) -> Dict[str, str]:
    env: Dict[str, str] = {}
    with open(path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            k, v = line.split('=', 1)
            k = k.strip()
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
                v = v[1:-1]
            env[k] = v
    return env


def ensure_secrets(w: WorkspaceClient, env: Dict[str, str], scope: str = 'agent_studio') -> None:
    # Create scope if missing
    scopes = list(w.secrets.list_scopes())
    # SDK returns a list of objects with attribute 'name'
    if not any(getattr(s, 'name', getattr(s, 'scope', '')) == scope for s in scopes):
        w.secrets.create_scope(scope)

    # Put the required secrets
    w.secrets.put_secret(scope=scope, key='open_ai', string_value=env['OPENAI_API_KEY'])
    w.secrets.put_secret(scope=scope, key='databricks_token', string_value=env['DATABRICKS_TOKEN'])
    w.secrets.put_secret(scope=scope, key='databricks_host', string_value=env['DATABRICKS_HOST'].rstrip('/'))
    w.secrets.put_secret(scope=scope, key='folder_path', string_value=env['AGENT_STUDIO_PATH'])


def select_or_start_cluster(w: WorkspaceClient) -> str:
    # Prefer a RUNNING all-purpose cluster; otherwise start the first available
    clusters = list(w.clusters.list())
    if not clusters:
        # Create a small all-purpose cluster if none exists
        spark_version = w.clusters.select_spark_version()
        node_type = w.clusters.select_node_type()
        created = w.clusters.create_and_wait(
            cluster_name="agent-studio-runner",
            spark_version=spark_version,
            node_type_id=node_type,
            autotermination_minutes=30,
            num_workers=1,
        )
        return created.cluster_id

    def state_of(c):
        # c.state may be enum-like or string; normalize
        return str(getattr(c, 'state', '') or '').upper()

    running = [c for c in clusters if state_of(c) == 'RUNNING']
    if running:
        return running[0].cluster_id

    # fallback: start the first cluster
    target = clusters[0]
    w.clusters.start_and_wait(cluster_id=target.cluster_id)
    c = w.clusters.get(cluster_id=target.cluster_id)
    return c.cluster_id


def http(host: str, token: str, method: str, path: str, payload: Optional[Any] = None):
    url = host.rstrip('/') + path
    data = None
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    if payload is not None:
        data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.getcode(), resp.read()
    except urllib.error.HTTPError as e:
        return e.getcode(), e.read()


def build_agent_and_endpoint(cluster_id: str, env: Dict[str, str]) -> str:
    # Import and run AgentCreator directly to submit a job that builds & deploys the agent
    sys.path.insert(0, os.getcwd())
    from Core.AgentCreator import AgentParser

    # Load agent YAML and align params with .env
    with open('Agent/yaml/main_default_sample_agent.yaml', 'r') as f:
        agent_spec = yaml.safe_load(f)

    # Ensure model/provider sane; default sample uses '4' mapping to GPT-4-turbo
    agent_spec['provider'] = agent_spec.get('provider', 'openai')
    agent_spec['model'] = agent_spec.get('model', '4')

    # Update tool parameters from .env if present
    wid = env.get('DATABRICKS_WAREHOUSE_ID')
    if wid and agent_spec.get('tools'):
        for t in agent_spec['tools']:
            if isinstance(t, dict) and t.get('parameters') and 'warehouse_id' in t['parameters']:
                t['parameters']['warehouse_id'] = wid

    # Re-serialize to YAML string
    agent_yaml = yaml.safe_dump(agent_spec, sort_keys=False)

    # Ensure required env vars for AgentCreator
    os.environ['DATABRICKS_HOST'] = env['DATABRICKS_HOST'].rstrip('/')
    os.environ['DATABRICKS_TOKEN'] = env['DATABRICKS_TOKEN']
    os.environ['AGENT_STUDIO_PATH'] = env['AGENT_STUDIO_PATH']

    parser = AgentParser(cluster_id)
    result = parser.create_agent(agent_yaml)
    return result


def wait_for_endpoint(w: WorkspaceClient, name: str, timeout_s: int = 900) -> Dict[str, Any]:
    start = time.time()
    last = None
    while time.time() - start < timeout_s:
        try:
            ep = w.serving_endpoints.get(name)
            last = ep.as_dict() if hasattr(ep, 'as_dict') else ep
            # Heuristic: status.state == 'READY'
            state = None
            conf = last.get('state') or last.get('config') or {}
            # different SDK versions shape differently; prefer state.ready
            state = last.get('state', {}).get('ready') or last.get('state', {}).get('deployment') or last.get('state', {}).get('state')
            # Alternatively, when not available, break on no 'pending_config'
            if last.get('state', {}).get('ready', '') == 'READY' or last.get('state', {}).get('config_update') == 'NOT_UPDATING':
                return last
        except Exception:
            pass
        time.sleep(10)
    return last or {}


def invoke_endpoint(host: str, token: str, endpoint: str, prompt: str) -> Dict[str, Any]:
    status, body = http(host, token, 'POST', f'/serving-endpoints/{endpoint}/invocations', {
        'messages': [
            {'role': 'user', 'content': prompt}
        ]
    })
    out: Dict[str, Any] = {
        'http_status': status,
    }
    try:
        out['response'] = json.loads(body.decode('utf-8'))
    except Exception:
        out['raw'] = body.decode('utf-8', 'ignore')
    return out


def run_via_local_mlflow(env: Dict[str, str], w: WorkspaceClient) -> None:
    import mlflow
    # Ensure repo root on sys.path for package imports
    repo_root = os.getcwd()
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from Core.DatabricksModel import register_model, create_endpoint

    # Configure MLflow to use Databricks backend and UC registry
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    # Optional: set an experiment under the user
    exp_user = env.get('DATABRICKS_EMAIL') or 'default'
    mlflow.set_experiment(f"/Users/{exp_user}/agent-studio")

    # Ensure environment for create_endpoint
    os.environ['OPENAI_API_KEY'] = env['OPENAI_API_KEY']
    os.environ['DATABRICKS_HOST'] = env['DATABRICKS_HOST'].rstrip('/')
    os.environ['DATABRICKS_TOKEN'] = env['DATABRICKS_TOKEN']

    # Log and register model using code-based (loader_module) approach to avoid pickling
    short_name = 'sample_agent'
    long_name = 'main.default.sample_agent'
    from mlflow.models import infer_signature
    sample_input = {
        'messages': [{'role': 'user', 'content': 'Hello'}]
    }
    sample_output = {
        'run_id': 'ABC123',
        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'response'}}],
        'thread': [{'role': 'user', 'content': 'Hello'}]
    }
    signature = infer_signature(sample_input, sample_output)

    pip_requirements = [
        'mlflow',
        'langchain',
        'langchain-core',
        'pydantic',
        'openai',
        'langchain-openai',
        'langchain-community',
        'uuid',
        'psutil',
        'databricks-sql-connector',
        'databricks-vectorsearch'
    ]

    with mlflow.start_run(run_name=f'{short_name}_{int(time.time())}') as run:
        mlflow.pyfunc.log_model(
            name='model',
            loader_module='Core.pyfunc_model',
            code_paths=['Core', 'Tool'],
            signature=signature,
            input_example=sample_input,
            metadata={"task": "llm/v1/chat"},
            pip_requirements=pip_requirements
        )
        run_id = run.info.run_id
    registered = register_model(run_id, long_name, 'prod')

    # Create or update serving endpoint with secret-based env vars
    ext_endpoint = env.get('DATABRICKS_EXTERNAL_MODEL') or 'databricks-dbrx-instruct'
    endpoint_cfg = {
        'environment_vars': {
            # Keep token/host for intra-workspace calls (ChatDatabricks & SQL tools)
            'DATABRICKS_TOKEN': '{{secrets/agent_studio/databricks_token}}',
            'DATABRICKS_HOST': '{{secrets/agent_studio/databricks_host}}',

            # Optional: still pass OPENAI key if external model uses it (not required for foundation models)
            'OPENAI_API_KEY': '{{secrets/agent_studio/open_ai}}',

            # Tool + chatbot config
            'DATABRICKS_WAREHOUSE_ID': env.get('DATABRICKS_WAREHOUSE_ID', ''),
            'CHATBOT_PROVIDER': 'databricks',
            'CHATBOT_ENDPOINT_TYPE': 'chat-basic',
            'CHATBOT_MODEL': ext_endpoint,
            'CHATBOT_CATALOG': 'main',
            'CHATBOT_SCHEMA': 'default',
            'CHATBOT_INSTRUCTION': 'You are a text to sql agent. Use tools to explore tables and answer questions.'
        }
    }
    ep = create_endpoint(short_name, registered, endpoint_cfg)
    print('[ok] Submitted serving endpoint create/update:', json.dumps(ep))


def main():
    env = parse_env('.env')

    for key in ['DATABRICKS_HOST', 'DATABRICKS_TOKEN', 'OPENAI_API_KEY', 'AGENT_STUDIO_PATH']:
        if not env.get(key):
            print(f"ERROR: Missing {key} in .env", file=sys.stderr)
            sys.exit(1)

    # Initialize SDK client
    w = WorkspaceClient(host=env['DATABRICKS_HOST'].rstrip('/'), token=env['DATABRICKS_TOKEN'])

    # 1) Ensure secrets are present
    ensure_secrets(w, env)
    print('[ok] Secrets ensured in scope: agent_studio')

    # 2) Deploy via MLflow locally (no cluster requirement)
    run_via_local_mlflow(env, w)

    # 3) Wait for serving endpoint to be ready
    endpoint_name = 'sample_agent'  # derived from YAML name main.default.sample_agent
    ep = wait_for_endpoint(w, endpoint_name)
    state = (ep or {}).get('state', {})
    print(f"[ok] Endpoint status: {json.dumps(state)}")

    # 4) Invoke endpoint with a simple prompt
    prompt = 'List the tables available in catalog main, schema default.'
    inv = invoke_endpoint(env['DATABRICKS_HOST'], env['DATABRICKS_TOKEN'], endpoint_name, prompt)
    print('[ok] Invocation result:')
    print(json.dumps(inv, indent=2))


if __name__ == '__main__':
    main()
