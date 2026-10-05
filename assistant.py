import os

import pandas as pd
import joblib

from google import genai
from google.genai import types


# =========================================================
# 1. MODEL CONFIGURATION
# =========================================================

# ถ้ามี .env อยู่ก็ยังสามารถใช้ GEMINI_MODEL ได้
# แต่ไม่จำเป็นต้องมี GEMINI_API_KEY แล้ว
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


# =========================================================
# 2. LOAD CUSTOMER SEGMENT DATA
# =========================================================

customer_segments = pd.read_csv(
    "data/customer_segments.csv"
)


# =========================================================
# 3. LOAD XGBOOST MODEL
# =========================================================

profit_model = joblib.load(
    "xgboost_profit_model.pkl"
)


# =========================================================
# 4. CUSTOMER SEGMENT FUNCTION
# =========================================================

def get_customer_segment(customer_id):

    customer = customer_segments[
        customer_segments["Customer ID"] == customer_id
    ]

    if customer.empty:
        return {
            "found": False,
            "customer_id": customer_id,
            "message": "Customer ID not found."
        }

    row = customer.iloc[0]

    return {
        "found": True,
        "customer_id": customer_id,
        "cluster": int(row["Cluster"])
    }


# =========================================================
# 5. PROFIT PREDICTION FUNCTION
# =========================================================

def predict_profit(
    sales,
    quantity,
    discount,
    category,
    sub_category,
    region,
    ship_mode,
    segment
):

    new_order = pd.DataFrame({

        "Sales": [sales],

        "Quantity": [quantity],

        "Discount": [discount],

        "Category": [category],

        "Sub-Category": [sub_category],

        "Region": [region],

        "Ship Mode": [ship_mode],

        "Segment": [segment]

    })

    prediction = profit_model.predict(
        new_order
    )

    predicted_profit = float(
        prediction[0]
    )

    return {
        "predicted_profit": round(
            predicted_profit,
            2
        )
    }


# =========================================================
# 6. TOOL DEFINITIONS
# =========================================================

get_customer_segment_declaration = (
    types.FunctionDeclaration(

        name="get_customer_segment",

        description=(
            "Find the customer cluster assigned "
            "by the K-Means customer segmentation model."
        ),

        parameters=types.Schema(

            type="OBJECT",

            properties={

                "customer_id": types.Schema(
                    type="STRING",
                    description=(
                        "Customer ID, for example AA-10315"
                    )
                )

            },

            required=[
                "customer_id"
            ]
        )
    )
)


predict_profit_declaration = (
    types.FunctionDeclaration(

        name="predict_profit",

        description=(
            "Predict the profit of a new order "
            "using the trained XGBoost regression model."
        ),

        parameters=types.Schema(

            type="OBJECT",

            properties={

                "sales": types.Schema(
                    type="NUMBER",
                    description="Sales amount"
                ),

                "quantity": types.Schema(
                    type="INTEGER",
                    description="Number of items"
                ),

                "discount": types.Schema(
                    type="NUMBER",
                    description=(
                        "Discount as decimal. "
                        "10% = 0.10"
                    )
                ),

                "category": types.Schema(
                    type="STRING",
                    description="Product category"
                ),

                "sub_category": types.Schema(
                    type="STRING",
                    description="Product sub-category"
                ),

                "region": types.Schema(
                    type="STRING",
                    description="Sales region"
                ),

                "ship_mode": types.Schema(
                    type="STRING",
                    description="Shipping mode"
                ),

                "segment": types.Schema(
                    type="STRING",
                    description="Customer segment"
                )

            },

            required=[

                "sales",
                "quantity",
                "discount",
                "category",
                "sub_category",
                "region",
                "ship_mode",
                "segment"

            ]
        )
    )
)


# =========================================================
# 7. GEMINI TOOLS
# =========================================================

tools = types.Tool(

    function_declarations=[

        get_customer_segment_declaration,

        predict_profit_declaration

    ]

)


# =========================================================
# 8. SYSTEM INSTRUCTION
# =========================================================

SYSTEM_INSTRUCTION = """

You are an AI Business Analyst Assistant.

You help users analyze Superstore business data.

You have access to two machine learning tools.

1. get_customer_segment
   Uses the project's K-Means customer segmentation model.

2. predict_profit
   Uses the project's trained XGBoost regression model.

IMPORTANT RULES:

- Never invent customer cluster results.

- Never calculate or guess predicted profit yourself.

- Always use the appropriate tool when the user asks
  about customer segmentation or profit prediction.

- If a question requires both tools,
  use both tools.

- Use previous conversation context when appropriate.

- If the user refers to "this customer",
  "that customer",
  "the same customer",
  or similar expressions,
  use the customer information from previous conversation.

- If required information is missing,
  ask the user for the missing information.

- Answer clearly in Thai.

- A predicted profit is a machine learning prediction,
  not a guaranteed actual profit.

- Do not claim that a prediction is actual historical profit.

"""


# =========================================================
# 9. EXECUTE TOOL
# =========================================================

def execute_function(
    function_name,
    arguments
):

    if function_name == "get_customer_segment":

        return get_customer_segment(

            customer_id=arguments[
                "customer_id"
            ]

        )


    elif function_name == "predict_profit":

        return predict_profit(

            sales=arguments[
                "sales"
            ],

            quantity=arguments[
                "quantity"
            ],

            discount=arguments[
                "discount"
            ],

            category=arguments[
                "category"
            ],

            sub_category=arguments[
                "sub_category"
            ],

            region=arguments[
                "region"
            ],

            ship_mode=arguments[
                "ship_mode"
            ],

            segment=arguments[
                "segment"
            ]

        )


    else:

        return {

            "error":
            f"Unknown function: {function_name}"

        }


# =========================================================
# 10. CREATE GEMINI CONFIG
# =========================================================

def create_gemini_config():

    return types.GenerateContentConfig(

        system_instruction=SYSTEM_INSTRUCTION,

        tools=[
            tools
        ],

        temperature=0.2

    )


# =========================================================
# 11. ASK ASSISTANT
# =========================================================

def ask_assistant(
    user_question,
    api_key,
    conversation_history=None
):

    # -----------------------------------------------------
    # Check API Key
    # -----------------------------------------------------

    if not api_key:

        return (
            "กรุณากรอก Gemini API Key "
            "ก่อนใช้งาน AI Assistant"
        )


    # -----------------------------------------------------
    # Create Gemini Client
    # -----------------------------------------------------

    client = genai.Client(
        api_key=api_key
    )


    # -----------------------------------------------------
    # Create conversation history
    # -----------------------------------------------------

    if conversation_history is None:

        conversation_history = []


    # -----------------------------------------------------
    # Gemini Configuration
    # -----------------------------------------------------

    config = create_gemini_config()


    # -----------------------------------------------------
    # Add user message
    # -----------------------------------------------------

    conversation_history.append(

        types.Content(

            role="user",

            parts=[

                types.Part(
                    text=user_question
                )

            ]

        )

    )


    # -----------------------------------------------------
    # First Gemini request
    # -----------------------------------------------------

    response = client.models.generate_content(

        model=MODEL_NAME,

        contents=conversation_history,

        config=config

    )


    # -----------------------------------------------------
    # No tool call
    # -----------------------------------------------------

    if not response.function_calls:

        conversation_history.append(

            response.candidates[0].content

        )

        return response.text


    # -----------------------------------------------------
    # Save Gemini tool-call response
    # -----------------------------------------------------

    conversation_history.append(

        response.candidates[0].content

    )


    # -----------------------------------------------------
    # Execute every requested tool
    # -----------------------------------------------------

    for function_call in response.function_calls:

        function_name = function_call.name

        arguments = dict(
            function_call.args
        )


        print()
        print(
            f"[Tool] {function_name}"
        )

        print(
            f"[Arguments] {arguments}"
        )


        # Execute Python function

        result = execute_function(

            function_name,

            arguments

        )


        print(
            f"[Result] {result}"
        )


        # -------------------------------------------------
        # Create function response
        # -------------------------------------------------

        function_response = types.Part(

            function_response=types.FunctionResponse(

                name=function_name,

                id=function_call.id,

                response={
                    "result": result
                }

            )

        )


        # -------------------------------------------------
        # Add result to conversation
        # -------------------------------------------------

        conversation_history.append(

            types.Content(

                role="user",

                parts=[

                    function_response

                ]

            )

        )


    # -----------------------------------------------------
    # Ask Gemini for final answer
    # -----------------------------------------------------

    final_response = client.models.generate_content(

        model=MODEL_NAME,

        contents=conversation_history,

        config=config

    )


    # -----------------------------------------------------
    # Save final answer
    # -----------------------------------------------------

    conversation_history.append(

        final_response.candidates[0].content

    )


    return final_response.text


# =========================================================
# 12. TERMINAL TEST
# =========================================================

if __name__ == "__main__":

    print()
    print("=" * 55)
    print(
        "        AI Business Analyst Assistant"
    )
    print("=" * 55)

    print(
        "Type 'exit' to quit."
    )

    print()

    api_key = input(
        "Enter Gemini API Key: "
    ).strip()


    conversation_history = []


    while True:

        question = input(
            "You: "
        )

        if question.lower() == "exit":

            print()
            print("Goodbye!")

            break


        try:

            answer = ask_assistant(

                question,

                api_key,

                conversation_history

            )

            print()
            print(
                "Assistant:",
                answer
            )

            print()

        except Exception as e:

            print()
            print(
                "Error:",
                e
            )

            print()
            