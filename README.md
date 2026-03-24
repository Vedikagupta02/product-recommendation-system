# E-commerce Product Recommendation System with Gen AI

This project is an end-to-end e-commerce product recommendation system that utilizes the power of natural language processing (NLP), Hybrid Machine Learning (Content-based & Collaborative Filtering), and RAG Techniques to provide personalized product recommendations. The system is built using Python and leverages libraries and frameworks such as Streamlit, Pandas, Scikit-learn, and Langchain with local Ollama models. 

Fundamentally different from purely traditional or purely LLM-based systems, this project uses a multi-stage hybrid approach: it retrieves candidate products using TF-IDF (Content-Based), re-ranks them based on user history (Collaborative Filtering), and then uses RAG with a local LLM (`llama3.2` via Ollama) to explain the recommendations based on user input and preferences. This ensures completely free and private execution with no external API costs.

## Features

- **Hybrid Recommendation Engine:** Combines Content-Based Filtering (TF-IDF) and Collaborative Filtering for robust product candidate generation and re-ranking.
- **Local GenAI Explanations:** Utilizes `Ollama` (Llama 3.2) to provide intelligent, natural language explanations for why specific products were recommended.
- **No API Costs:** Fully local execution. No OpenAI API keys or paid subscriptions are required.
- **Interactive UI:** The system features an intuitive and user-friendly interface built with Streamlit, allowing users to input preferences (department, category, brand, price) and an optional User ID.
- **Data Analysis & Processing:** Comprehensive data preprocessing on the e-commerce dataset for optimal ML performance.

## Installation

Clone the repository:
```bash
git clone https://github.com/Vedikagupta02/product-recommendation-system.git
```

Navigate to the project directory:
```bash
cd product-recommendation-system/Ecommerce_recommendation_using_GenAI
```

Install the required Python dependencies:
```bash
pip install -r requirements.txt
```

Set up Ollama (Local LLM):
1. Install **[Ollama](https://ollama.com)** on your system and make sure it is running.
2. Pull the required model by running:
```bash
ollama pull llama3.2
```

*(Optional)* You can customize the model by creating a `.env` file in the project root directory and setting `OLLAMA_MODEL`:
```
OLLAMA_MODEL=llama3.2
```

## Usage

- Prepare your e-commerce dataset in CSV format and place it in the project directory (default: `flipkart_com-ecommerce_sample.csv`).
- Run the Streamlit app:
```bash
streamlit run app.py
```
- Access the application through the provided URL in your web browser.
- Enter your product preferences and an optional User ID (e.g., 0–799) to enable the collaborative re-ranking step.
- Click "Get Recommendations" to view the Hybrid ML shortlist and the RAG-generated explanation.

## Dataset

The project utilizes the Flipkart e-commerce dataset, which contains information about various products sold on the Flipkart platform. The dataset includes details such as product name, description, category, price, brand, and more. You can replace the dataset with your own e-commerce dataset in CSV format.

## System Setup & Initialization

Instead of managing a persistent vector database, this system dynamically loads the dataset and trains the hybrid recommenders (content-based and collaborative models) on startup. This is a one-time process per session, caching the results in memory via Streamlit for lightning-fast subsequent queries. 

## Project Structure

```
ecommerce-product-recommendation/
├── app.py
├── data_processing.py
├── recommendation.py
├── ml_models/
│   ├── collaborative.py
│   ├── content_based.py
│   └── hybrid.py
├── requirements.txt
├── .env
└── README.md
```
- `app.py`: The main Streamlit application file that handles the user interface.
- `data_processing.py`: Contains functions for data preprocessing, cleaning, and analysis.
- `recommendation.py`: The main entry point for the hybrid setup and Ollama inference.
- `ml_models/`: Contains the modular implementation of Content-Based, Collaborative, and Hybrid recommendation systems.
- `requirements.txt`: Lists the required Python dependencies.
- `.env`: Environment file to optionally store configuration (like the Ollama model name).

## Dependencies

The project relies on the following major dependencies:
- **Streamlit**: For building the interactive user interface.
- **Pandas & NumPy**: For data manipulation and arrays.
- **Scikit-learn**: For TF-IDF and collaborative filtering modeling.
- **Langchain & Langchain-Ollama**: For prompt orchestration and local LLM integration.

For a complete list of dependencies, please refer to the `requirements.txt` file.

## Future Enhancements

- Implement real user authentication and persistent user profiles.
- Expand the collaborative filtering dataset with real-world interactions.
- Optimize the initialization time for exceptionally large-scale datasets.
- Explore advanced NLP techniques and fine-tuned LLMs for enhanced accuracy.

## Contributing

Contributions to the project are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request. Make sure to follow the project's code of conduct.

## License

This project is licensed under the MIT License.
