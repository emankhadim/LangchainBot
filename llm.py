
import os
import streamlit as st
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableSequence
import requests
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
car_api_key = os.getenv("CAR_API_KEY")

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = openai_api_key

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["car_info"],
    template="You are an expert on cars. Provide detailed information about the following car: {car_info}."
)

# Create the LLM instance
llm = OpenAI(api_key=openai_api_key)

# Create the RunnableSequence using the | operator to chain them
runnable_sequence = prompt_template | llm

# Function to fetch car information
def get_car_info(make, year, model=None):
    url = f"https://api.api-ninjas.com/v1/cars?make={make}&year={year}"
    if model:
        url += f"&model={model}"
    headers = {'X-Api-Key': car_api_key}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data:
            car_info_list = []
            for car_data in data[:10]:  # Limit to 10 cars for readability
                car_info_parts = {
                    "Make": car_data.get('make', 'N/A'),
                    "Model": car_data.get('model', 'N/A'),
                    "Year": car_data.get('year', 'N/A'),
                    "Transmission": car_data.get('transmission', 'N/A'),
                    "Cylinders": car_data.get('cylinders', 'N/A'),
                    "Fuel Type": car_data.get('fuel_type', 'N/A'),
                    "City MPG": car_data.get('city_mpg', 'N/A'),
                    "Highway MPG": car_data.get('highway_mpg', 'N/A'),
                    "Combined MPG": car_data.get('combination_mpg', 'N/A'),
                    "Class": car_data.get('class', 'N/A')
                }
                car_info_list.append(car_info_parts)
            return car_info_list
        else:
            return "Car information not available"
    else:
        return "Failed to retrieve car information"

# Streamlit app
st.title("LangChain Car Information Bot")

make = st.text_input("Enter car make:", "BMW")
model = st.text_input("Enter car model (optional):", "")
year = st.text_input("Enter car year:", "2020")

if st.button("Get Car Information"):
    if make and year:
        car_info_list = get_car_info(make, year, model)
        if isinstance(car_info_list, list):
            for car_info in car_info_list:
                st.write("### Car Information:")
                for key, value in car_info.items():
                    st.write(f"**{key}:** {value}")
                st.write("---")  # Separator between different cars
            try:
                response = runnable_sequence.invoke({"car_info": str(car_info_list[0])})  # Use the first car info for LLM
                st.write(f"### LLM Response:")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.write(car_info_list)
    else:
        st.write("Please enter the car make and year.")