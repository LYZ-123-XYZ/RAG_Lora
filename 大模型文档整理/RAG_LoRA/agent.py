import dotenv
dotenv.load_dotenv()

from llama_index.core import VectorStoreIndex, Document
from llama_index.core.agent import FnAgentWorker
from llama_index.core import PromptTemplate
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_agents import (
    AgentService,
    ControlPlaneServer,
    SimpleMessageQueue,
    PipelineOrchestrator,
    ServiceComponent,
)
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    PromptTemplate,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_agents.launchers import LocalLauncher
from llama_index.llms.huggingface import HuggingFaceLLM
import logging
# 修改为使用本地Llama3.1的相关设置
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoTokenizer, AutoModelForCausalLM

# change logging level to enable or disable more verbose logging
logging.getLogger("llama_agents").setLevel(logging.INFO)
embed_model = HuggingFaceEmbedding(model_name="bge-m3")

Settings.llm = None
Settings.embed_model = embed_model
# Load and index your document

documents = SimpleDirectoryReader("archinfo/").load_data()
index = VectorStoreIndex.from_documents(documents)

# Define a query rewrite agent
HYDE_PROMPT_STR = (
    "Please rewrite the following query to include more detail:\n{query_str}\n"
)
HYDE_PROMPT_TMPL = PromptTemplate(HYDE_PROMPT_STR)

llama3_1_model_path = "Llama3.1"  
tokenizer = AutoTokenizer.from_pretrained(llama3_1_model_path)
model = AutoModelForCausalLM.from_pretrained(llama3_1_model_path)

llm = HuggingFaceLLM(
    tokenizer=tokenizer,
    model=model,
)

def run_hyde_fn(state):
    prompt_tmpl, llm, input_str = (
        state["prompt_tmpl"],
        state["llm"],
        state["__task__"].input,
    )
    qp = QueryPipeline(chain=[prompt_tmpl, llm])
    output = qp.run(query_str=input_str)
    state["__output__"] = str(output)
    return state, True

hyde_agent = FnAgentWorker(
    fn=run_hyde_fn,
    initial_state={"prompt_tmpl": HYDE_PROMPT_TMPL, "llm": llm}
).as_agent()

# Define a RAG agent
def run_rag_fn(state):
    retriever, llm, input_str = (
        state["retriever"],
        state["llm"],
        state["__task__"].input,
    )
    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)
    response = query_engine.query(input_str)
    state["__output__"] = str(response)
    return state, True

rag_agent = FnAgentWorker(
    fn=run_rag_fn,
    initial_state={"retriever": index.as_retriever(), "llm": llm}
).as_agent()

# Set up the multi-agent system
message_queue = SimpleMessageQueue()

query_rewrite_service = AgentService(
    agent=hyde_agent,
    message_queue=message_queue,
    description="Query rewriting service",
    service_name="query_rewrite",
)

rag_service = AgentService(
    agent=rag_agent,
    message_queue=message_queue,
    description="RAG service",
    service_name="rag",
)

# Create the pipeline
pipeline = QueryPipeline(chain=[
    ServiceComponent.from_service_definition(query_rewrite_service),
    ServiceComponent.from_service_definition(rag_service),
])
orchestrator = PipelineOrchestrator(pipeline)

control_plane = ControlPlaneServer(
    message_queue=message_queue,
    orchestrator=orchestrator,
)

# Set up the launcher
launcher = LocalLauncher(
    [query_rewrite_service, rag_service],
    control_plane,
    message_queue,
)

# Run a query
result = launcher.launch_single("What is the architecture of H100?")
print(result)