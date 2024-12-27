import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# Simple App to recommend list of dishes based on ingredients provided by user

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq Chat Model
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# FastAPI app instance
app = FastAPI()

# System message setup
system_message = SystemMessage(
    content="""
    You are a helpful recipe recommendation assistant. Users will provide you with a list of ingredients they have.
    Your task is to recommend recipes that can be made using the provided ingredients. If no recipes can be found,
    suggest general tips or simple dishes they can make with their ingredients.
    """
)


# Request model
class IngredientsRequest(BaseModel):
    ingredients: list[str]  # Array of strings


# Response model
class RecipeResponse(BaseModel):
    recipes: str  # A string containing recipe suggestions


# Endpoint for recipe recommendations
@app.post("/recommend", response_model=RecipeResponse)
async def recommend_recipes(request: IngredientsRequest):
    # Ensure ingredients are provided
    if not request.ingredients:
        raise HTTPException(status_code=400, detail="Ingredients list cannot be empty")
    # Join ingredients into a comma-separated string
    ingredients_str = ", ".join(request.ingredients)

    # Human message for the model
    user_message = HumanMessage(
        content=f"I have these ingredients: {ingredients_str}. Can you recommend some recipes?"
    )

    # Get model response
    response = model.invoke([system_message, user_message])

    # Return the model's output as a response
    return {"recipes": response.content}
