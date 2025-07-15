from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
   memory_key="chat_history",
   input_key="input",
   return_messages=True
)