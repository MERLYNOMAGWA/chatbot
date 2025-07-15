from fastapi import APIRouter, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR
from App.agents.agent import create_agent
from App.memory import manager as memory

router = APIRouter()
history_router = APIRouter()
agent = create_agent()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

class ErrorResponse(BaseModel):
    error: str

@router.post("/chat", response_model=ChatResponse, responses={429: {"model": ErrorResponse}})
async def chat(request: Request, chat_req: ChatRequest):
    limiter = request.app.state.limiter
    await limiter.limit("5/minute")(lambda r: None)(request)

    try:
        user_input = chat_req.message
        print(f"User input: {user_input}")

        response = agent.run(user_input)

        print(f"Agent response: {response}")
        return {"response": response}

    except Exception as e:
        return JSONResponse(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": f"Internal server error: {str(e)}"},
        )

@history_router.get("/chat/history/{session_id}")
def get_chat_history(session_id: str):
    try:
        return {"session_id": session_id, "history": memory.buffer}
    except Exception as e:
        return {"error": str(e)}