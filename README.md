# LangChain Car Information Bot

## Overview

This project is a Streamlit app that uses LangChain and API Ninjas to fetch and display detailed information about cars based on the make and year. Model input is optional.

## Features

- Fetch car information from the API Ninjas Cars API.
- Display up to 10 cars' details in a structured, readable format.
- Handle missing or optional inputs gracefully.
- Use LangChain to provide expert-level car information.

## Technology Stack

- **Streamlit**: For creating the interactive web app.
- **LangChain**: For integrating language models to provide detailed car information.
- **API Ninjas**: For fetching car data.
- **OpenAI**: For language model integration.
- **Python**: Main programming language.
Getting Started

### Prerequisites

- Python 3.7+
- Streamlit
- Requests
- LangChain
- OpenAI
- dotenv

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/emankhadim/LangchainBot.git
   cd LangchainBot
   
2. Install the required packages:
 ```bash
   pip install -r requirements.txt
   
3. Create a .env file and add your API keys:

   OPENAI_API_KEY=your_openai_api_key
   CAR_API_KEY=your_car_api_key

4. Run the app:
  ```bash
   streamlit run app.py
 
