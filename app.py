# Import the required libraries
import streamlit as st
from playwright.sync_api import sync_playwright
import pandas as pd
import platform
import asyncio
import os
from openai import OpenAI
from scrapegraphai.graphs import SmartScraperGraph

if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Set up the Streamlit app
st.title("IPL Web Scraping AI Agent 🏏")
st.caption("This app scrapes IPL data from ESPN Cricinfo using OpenAI API")

# Get OpenAI API key from user
openai_access_token = st.text_input("OpenAI API Key", type="password")

if openai_access_token:
    os.environ["OPENAI_API_KEY"] = openai_access_token
    client = OpenAI(api_key=openai_access_token)

    model = st.radio(
        "Select the model",
        ["gpt-3.5-turbo", "gpt-4"],
        index=0,
    )
    graph_config = {
        "llm": {
            "api_key": openai_access_token,
            "model": model,
        },
    }

    # Default URL to IPL page on ESPN Cricinfo
    url = st.text_input("Enter the URL of the website you want to scrape", "https://www.espncricinfo.com/records/tournament/indian-premier-league-2024-15940")
    
    # Get the user prompt for IPL data
    user_prompt = st.text_input("What IPL data do you want to scrape from ESPN Cricinfo?", "Get the latest IPL match results and team standings.")

    # Create a SmartScraperGraph object
    smart_scraper_graph = SmartScraperGraph(
        prompt=user_prompt,
        source=url,
        config=graph_config
    )

    # Custom scrape function
    def scrape_with_playwright(url):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url)

            # Click on the "Most runs" link and wait for the table to load
            page.click('text="Most runs"')
            page.wait_for_selector("table", timeout=60000)
            page.wait_for_timeout(5000)  # Extra wait time for stability
            batting_content = page.content()

            # Navigate back to the main page or click on a direct link for "Most wickets"
            page.goto(url)
            page.wait_for_selector('text="Most wickets"', timeout=60000)
            page.click('text="Most wickets"')
            page.wait_for_selector("table", timeout=60000)
            page.wait_for_timeout(5000)  # Extra wait time for stability
            bowling_content = page.content()

            browser.close()

            return batting_content, bowling_content

    # Extract data from the HTML content
    def extract_data(html_content):
        dfs = pd.read_html(html_content)
        return dfs

    # Send data to OpenAI for summarization or insights
    def analyze_data_with_openai(dataframes, prompt):
        combined_data = "\n\n".join([df.to_csv(index=False) for df in dataframes])
        messages = [
            {
                "role": "system",
                "content": "You are a data analyst specialized in cricket statistics."
            },
            {
                "role": "user",
                "content": f"{prompt}\n\nHere is the data in CSV format:\n{combined_data}"
            }
        ]
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model
        )
        return chat_completion.choices[0].message.content

    # Scrape the website
    if st.button("Scrape"):
        try:
            batting_content, bowling_content = scrape_with_playwright(url)

            # Extract Batting Data
            batting_dataframes = extract_data(batting_content)
            for i, df in enumerate(batting_dataframes):
                st.write(f"Most Runs Table {i + 1}")
                st.write(df)
                csv = df.to_csv(index=False)
                st.download_button(
                    label=f"Download Most Runs Table {i + 1} as CSV",
                    data=csv,
                    file_name=f"most_runs_table_{i + 1}.csv",
                    mime="text/csv"
                )

            # Extract Bowling Data
            bowling_dataframes = extract_data(bowling_content)
            for i, df in enumerate(bowling_dataframes):
                st.write(f"Most Wickets Table {i + 1}")
                st.write(df)
                csv = df.to_csv(index=False)
                st.download_button(
                    label=f"Download Most Wickets Table {i + 1} as CSV",
                    data=csv,
                    file_name=f"most_wickets_table_{i + 1}.csv",
                    mime="text/csv"
                )

            # Analyze data with OpenAI
            if user_prompt:
                all_dataframes = batting_dataframes + bowling_dataframes
                analysis_result = analyze_data_with_openai(all_dataframes, user_prompt)
                st.write("Analysis Result:")
                st.write(analysis_result)

        except Exception as e:
            st.error(f"Error occurred during scraping: {e}")
