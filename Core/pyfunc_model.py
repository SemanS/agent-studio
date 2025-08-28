import os
from Core.DatabricksModel import ChatModel
from Tool.TextToSQL import UnityCatalog_Schema


class _PyfuncWrapper:
    def __init__(self, model: ChatModel):
        self._model = model

    # MLflow will call predict(model_input, params=None)
    def predict(self, model_input, params=None):
        # ChatModel.predict signature: (context, model_input, params)
        return self._model.predict(None, model_input, params)


def load_pyfunc(context):
    # Read configuration from environment variables set in Serving
    provider = os.environ.get('CHATBOT_PROVIDER', 'openai')
    model = os.environ.get('CHATBOT_MODEL', 'gpt-4o-mini')
    instruction = os.environ.get('CHATBOT_INSTRUCTION', 'You are a text to sql agent. Use tools to explore tables and answer questions.')
    endpoint_type = os.environ.get('CHATBOT_ENDPOINT_TYPE', 'chat-basic')

    warehouse_id = os.environ.get('DATABRICKS_WAREHOUSE_ID')
    catalog = os.environ.get('CHATBOT_CATALOG', 'main')
    schema = os.environ.get('CHATBOT_SCHEMA', 'default')

    helpers = []
    if warehouse_id:
        helpers.append(UnityCatalog_Schema(warehouse_id=warehouse_id, catalog=catalog, schema=schema))

    cfg = {
        'provider': provider,
        'model': model,
        'instruction_prompt': instruction,
        'endpoint_type': endpoint_type,
    }
    m = ChatModel(cfg, helpers)
    # Initialize internal bot and resources
    m.load_context(None)
    return _PyfuncWrapper(m)


# Compatibility: Some MLflow runtimes look for `_load_pyfunc`
def _load_pyfunc(context):
    return load_pyfunc(context)
