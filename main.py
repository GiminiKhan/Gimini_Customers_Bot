# import os 

# from dotenv import load_dotenv
# from typing import cast
# import chainlit as cl
# from agents import Agent, Runner , AsyncOpenAI, OpenAIChatCompletionsModel
# from agents.run import RunConfig

# load_dotenv()
# gemini_api_key = os.getenv("GEMINI_API_KEY")
# #check if the APi Key is present : if not raise an error 
# if not gemini_api_key:
#     raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# @cl.on_chat_start
# async def start():

#     #Reference: https://ai.google.dev/gemini-api/docs/openai
#     external_client = AsyncOpenAI(
#         api_key=gemini_api_key,
#         base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
#     )

#     model = OpenAIChatCompletionsModel(
#         model="gemini-2.5-flash",
#         openai_client=external_client
#     )

#     config = RunConfig(
#         model=model,
#         model_provider=external_client,
#         tracing_disabled=True
#     )

#     cl.user_session.set("chat history", [])
#     cl.user_session.set("config", config)

#     agent: Agent = Agent (name= "Assistant", instructions = "You are a helpful Assistant", model=model)

#     cl.user_session.set("agent",agent)
#     await cl.Message(content= "Welcome to the Penaversity AI Assistant ! How can i help you today?").send()

#     @cl.on_message
#     async def main (message: cl.Message):
#         msg = cl.Message(content= "Thinking...")
#         await msg.send()
#         agent: Agent = cast (Agent, cl.user_session.get("agent"))
#         config: RunConfig = cast (RunConfig, cl.user_session.get("config"))

#         history = cl.user_session.get("chat history") or []
#         history.append({"role": "user", "content": message.content})

#         try:
#             print("\n[calling agent with context]\n", history, "\n")

#             result = Runner.run_sync(
#                 starting_agent = agent,
#                 input = history,
#                 run_config=config
#             )

#             response_content =  result.final_output
#             msg.content = response_content
#             await msg.update()


#             cl.user_session.set("chat history", result.to_input_list())

#             print(f"User: {message.content}")
#             print(f"Assistant: {response_content}")

#         except Exception as e:
#             msg.content =f"Error : {str(e)}"
#             await msg.update()
#             print(f"Error: {str(e)}")



import os
from dotenv import load_dotenv
from typing import cast
import chainlit as cl

from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool, guardrail
from agents.run import RunConfig


# ------------------ ENV + API Key ------------------
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# ------------------ Function Tool ------------------
@function_tool
def get_order_status(order_id: str, query: str = "") -> str:
    """
    Simulated order lookup tool.
    Manual handling of is_enabled + error_function because v0.2.8 does not support them.
    """
    # manual is_enabled
    if "order" not in query.lower():
        return "⚠️ Order status lookup is not enabled for this query."

    fake_orders = {
        "123": "Shipped",
        "456": "Processing",
        "789": "Delivered"
    }

    status = fake_orders.get(order_id)
    if not status:
        # manual error_function
        return f"❌ Sorry, no order found with ID {order_id}. Please double-check!"
    return f"✅ Order {order_id} is currently {status}."

# ------------------ Guardrail ------------------
def guardrail_check(user_input: str) -> bool:
    """Block negative or offensive language."""
    negative_keywords = ["bad", "stupid", "hate", "useless"]
    return any(word in user_input.lower() for word in negative_keywords)

# ------------------ Chainlit Start ------------------
@cl.on_chat_start
async def start():
    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    model = OpenAIChatCompletionsModel(
        model="gemini-2.5-flash",
        openai_client=external_client
    )

    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True
    )

    cl.user_session.set("chat history", [])
    cl.user_session.set("config", config)

    # Agents
    bot_agent: Agent = Agent(
        name="BotAgent",
        instructions="You are a helpful customer support assistant. "
                     "Answer FAQs, check order status, or escalate if needed.",
        model=model,
        tools=[get_order_status]
    )

    human_agent: Agent = Agent(
        name="HumanAgent",
        instructions="You are a friendly human support agent. Take over if bot cannot handle.",
        model=model
    )

    cl.user_session.set("bot_agent", bot_agent)
    cl.user_session.set("human_agent", human_agent)

    await cl.Message(content="🤖 Welcome to Customer Support! How can I help you today?").send()

# ------------------ On Message ------------------
@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content="🤔 Thinking...")
    await msg.send()

    bot_agent: Agent = cast(Agent, cl.user_session.get("bot_agent"))
    human_agent: Agent = cast(Agent, cl.user_session.get("human_agent"))
    config: RunConfig = cast(RunConfig, cl.user_session.get("config"))

    history = cl.user_session.get("chat history") or []
    history.append({"role": "user", "content": message.content})

    try:
        # Guardrail check
        if guardrail_check(message.content):
            response = "⚠️ Please keep the conversation respectful. Let's try again!"
            msg.content = response
            await msg.update()
            return

        print("\n[Calling BotAgent with context]\n", history, "\n")

        # Run bot
        result = Runner.run_sync(
            starting_agent=bot_agent,
            input=history,
            run_config=config
        )

        response_content = result.final_output

        # Escalation: if bot fails / responds with confusion
        if "I don't know" in response_content or "escalate" in message.content.lower():
            print("[Escalating to HumanAgent]")
            result = Runner.run_sync(
                starting_agent=human_agent,
                input=history,
                run_config=config
            )
            response_content = result.final_output

        msg.content = response_content
        await msg.update()

        # Save chat history
        cl.user_session.set("chat history", result.to_input_list())

        # Logging
        print(f"👤 User: {message.content}")
        print(f"🤖 Assistant: {response_content}")

    except Exception as e:
        msg.content = f"❌ Error: {str(e)}"
        await msg.update()
        print(f"Error: {str(e)}")
