import os
import pandas as pd
import joblib

from google import genai
from google.genai import types


# =========================================================
# 1. MODEL CONFIGURATION
# =========================================================

# สามารถเปลี่ยน model ผ่าน environment variable ได้
MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)


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
# 4. CUSTOMER CLUSTER PROFILES
# =========================================================

CLUSTER_PROFILES = {

    0: {
        "name": "High Value Customer",
        "description": (
            "ลูกค้าที่มีมูลค่าการซื้อสูง "
            "ซื้อค่อนข้างบ่อย และสร้างกำไรสูง"
        ),
        "characteristics": {
            "Recency": "ประมาณ 126 วัน",
            "Frequency": "ประมาณ 7.6 ครั้ง",
            "Monetary": "ประมาณ $8,112",
            "Total Profit": "ประมาณ $1,687",
            "Average Discount": "ประมาณ 12%"
        }
    },

    1: {
        "name": "Regular Customer",
        "description": (
            "ลูกค้าที่มีพฤติกรรมการซื้อค่อนข้างสม่ำเสมอ "
            "มีมูลค่าการซื้อและกำไรอยู่ในระดับปานกลาง"
        ),
        "characteristics": {
            "Recency": "ประมาณ 77 วัน",
            "Frequency": "ประมาณ 6.7 ครั้ง",
            "Monetary": "ประมาณ $2,396",
            "Total Profit": "ประมาณ $195",
            "Average Discount": "ประมาณ 16%"
        }
    },

    2: {
        "name": "Inactive / At-Risk Customer",
        "description": (
            "ลูกค้าที่ไม่ได้ซื้อมานาน "
            "ซื้อไม่บ่อย และมีมูลค่าการซื้อค่อนข้างต่ำ"
        ),
        "characteristics": {
            "Recency": "ประมาณ 484 วัน",
            "Frequency": "ประมาณ 3.7 ครั้ง",
            "Monetary": "ประมาณ $1,219",
            "Total Profit": "ประมาณ $116",
            "Average Discount": "ประมาณ 18%"
        }
    }
}


# =========================================================
# 5. CUSTOMER SEGMENT FUNCTION
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

    cluster = int(row["Cluster"])

    profile = CLUSTER_PROFILES.get(
        cluster,
        {
            "name": "Unknown",
            "description": "ไม่พบคำอธิบายของ Cluster นี้",
            "characteristics": {}
        }
    )

    return {
        "found": True,
        "customer_id": customer_id,
        "cluster": cluster,
        "cluster_name": profile["name"],
        "description": profile["description"],
        "characteristics": profile["characteristics"]
    }


# =========================================================
# 6. GET CLUSTER PROFILE
# =========================================================

def get_cluster_profile(cluster):

    try:
        cluster = int(cluster)
    except (ValueError, TypeError):
        return {
            "found": False,
            "message": "Cluster must be 0, 1, or 2."
        }

    if cluster not in CLUSTER_PROFILES:
        return {
            "found": False,
            "cluster": cluster,
            "message": "Cluster must be 0, 1, or 2."
        }

    profile = CLUSTER_PROFILES[cluster]

    return {
        "found": True,
        "cluster": cluster,
        "cluster_name": profile["name"],
        "description": profile["description"],
        "characteristics": profile["characteristics"]
    }


# =========================================================
# 7. PROFIT PREDICTION FUNCTION
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
# 8. TOOL DECLARATION
# =========================================================

get_customer_segment_declaration = (
    types.FunctionDeclaration(
        name="get_customer_segment",

        description=(
            "Find the customer cluster assigned by "
            "the K-Means customer segmentation model. "
            "Also returns the cluster name, description, "
            "and customer characteristics."
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


# =========================================================
# 9. CLUSTER PROFILE TOOL
# =========================================================

get_cluster_profile_declaration = (
    types.FunctionDeclaration(
        name="get_cluster_profile",

        description=(
            "Get the meaning and characteristics of a "
            "customer cluster from the K-Means segmentation model. "
            "Use this tool when the user asks what Cluster 0, "
            "Cluster 1, or Cluster 2 represents."
        ),

        parameters=types.Schema(
            type="OBJECT",

            properties={
                "cluster": types.Schema(
                    type="INTEGER",
                    description=(
                        "Customer cluster number: 0, 1, or 2"
                    )
                )
            },

            required=[
                "cluster"
            ]
        )
    )
)


# =========================================================
# 10. PROFIT PREDICTION TOOL DECLARATION
# =========================================================

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
# 11. GEMINI TOOLS
# =========================================================

tools = types.Tool(
    function_declarations=[
        get_customer_segment_declaration,
        get_cluster_profile_declaration,
        predict_profit_declaration
    ]
)


# =========================================================
# 12. SYSTEM INSTRUCTION
# =========================================================

SYSTEM_INSTRUCTION = """

You are an AI Business Analyst Assistant.

You help users analyze Superstore business data.

You have access to three machine learning tools.

1. get_customer_segment
   Uses the project's K-Means customer segmentation model
   to find a customer's cluster.

2. get_cluster_profile
   Explains what each customer cluster means and describes
   the characteristics of that cluster.

3. predict_profit
   Uses the project's trained XGBoost regression model
   to predict profit for a new order.


IMPORTANT RULES:

- Never invent customer cluster results.

- Always use get_customer_segment when the user asks
  which cluster a specific customer belongs to.

- Always use get_cluster_profile when the user asks
  what a cluster means or what type of customer belongs
  to a particular cluster.

- If the user asks:
  "Cluster 1 เป็นอะไร"
  "Cluster 1 คืออะไร"
  "กลุ่ม 1 เป็นลูกค้าแบบไหน"
  "ลูกค้ากลุ่มนี้มีลักษณะอย่างไร"

  use get_cluster_profile.

- If the user asks:
  "AA-10315 อยู่กลุ่มไหน"

  use get_customer_segment.

- If the user asks about both a customer and the meaning
  of that customer's cluster, use both tools if necessary.

- Never calculate or guess predicted profit yourself.

- Always use predict_profit when the user asks
  about predicted profit.

- If a question requires multiple tools,
  use all appropriate tools.

- Use previous conversation context when appropriate.

- If the user refers to:
  "this customer",
  "that customer",
  "the same customer"

  use the customer information from previous conversation.

- If required information is missing,
  ask the user for the missing information.

- Answer clearly in Thai.

- Explain cluster names in Thai when appropriate.

- A predicted profit is a Machine Learning prediction,
  not a guaranteed actual profit.

- Do not claim that a prediction is actual historical profit.

"""


# =========================================================
# 13. EXECUTE TOOL
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

    elif function_name == "get_cluster_profile":

        return get_cluster_profile(
            cluster=arguments[
                "cluster"
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
            "error": (
                f"Unknown function: "
                f"{function_name}"
            )
        }


# =========================================================
# 14. CREATE GEMINI CONFIG
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
# 15. ASK ASSISTANT
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
# 16. TERMINAL TEST
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
            