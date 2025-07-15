from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel
from App.retriever import create_retriever

retriever = create_retriever()

class MenuSearchInput(BaseModel):
    query: str

class MenuSearchTool(BaseTool):
    name: str = "Menu Search Tool"
    description: str = (
        "Search for information about menu items, open hours, reservations, location, or student discounts. "
        "Use this when the user asks questions like 'Do you have vegan dishes?', "
        "'What time do you open?', 'Where are you located?', or 'Do students get a discount?'."
    )
    args_schema: Type[BaseModel] = MenuSearchInput

    def _run(self, query: str) -> str:
        try:
            results = retriever.get_relevant_documents(query)
            if not results:
                return "Sorry, I couldn't find an answer related to that."
            return "\n".join(doc.page_content for doc in results)
        except Exception as e:
            return f"Error during search: {str(e)}"

    def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not supported for MenuSearchTool.")