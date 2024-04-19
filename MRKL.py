from langchain_cohere.llms import Cohere
from langchain_cohere import ChatCohere
from langchain.chains import LLMMathChain
#from langchain.utilities.serpapi import SerpAPIWrapper
from langchain.agents import initialize_agent, Tool, AgentExecutor
import os
import chainlit as cl
import cohere
from duckduckgo_search import DDGS 

#os.environ["SERPAPI_API_KEY"] = "31f445af52d97d700dd2ad3a81d9dc588716462979ecd9078424f289e5d42fc0"

@cl.on_chat_start
def start():
    llm = ChatCohere(cohere_api_key ="v1RU5Wj3wJaMCVXGda36RfM7dW4xNv7dkt4qBekK")
    #llm1 = Cohere(cohere_api_key ="v1RU5Wj3wJaMCVXGda36RfM7dW4xNv7dkt4qBekK",temperature=0, streaming=True)
    search = DDGS()
    #llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

    tools = [
        Tool(
            name="Search",
            func=search.text,
            description="useful for when you need to answer questions about current events. You should ask targeted questions",
        ),
        Tool(
            name="Calculator",
             func=llm.invoke,
            description="useful for when you need to answer questions about math",
        ),
    ]
    agent = initialize_agent(
        tools, llm, agent="chat-zero-shot-react-description", verbose=True
    )
    cl.user_session.set("agent", agent)

@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)
    # Process the message content asynchronously
    result = await cl.make_async(agent.run)(message.content, callbacks=[cb])
    # Send the processed message back to the user
    await cl.Message(content=result).send()