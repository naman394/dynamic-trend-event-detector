try:
    from bertopic import BERTopic
    import pandas as pd
    import os

    # Load data
    df = pd.read_csv('data/news_headlines.csv')

    docs = df['headline_text'].tolist()

    # Initialize BERTopic
    # Note: This will download a pre-trained SBERT model (e.g., all-MiniLM-L6-v2)
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(docs)

    # Get topic info
    info = topic_model.get_topic_info()
    print("BERTopic Information:")
    print(info.head(10))

    # Save visualization
    if not os.path.exists('reports'):
        os.makedirs('reports')
    
    fig = topic_model.visualize_topics()
    fig.write_html("reports/bertopic_viz.html")
    print("Saved BERTopic interactive visualization to reports/bertopic_viz.html")

except ImportError:
    print("BERTopic not installed. Please run 'pip install bertopic' in the virtual environment.")
except Exception as e:
    print(f"An error occurred in BERTopic implementation: {e}")
