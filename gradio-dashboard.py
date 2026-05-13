import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr
from matplotlib.image import thumbnail

load_dotenv()

books = pd.read_csv("books_with_emotion.csv")

#### THUMBNAILS - from thumbnails URL, return a large resolution image to use as book cover visual
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
# if no cover, use cover-not-found img
books["large_thumbnail"] = np.where(books["large_thumbnail"].isna(),
                                    "cover-not-found.jpg",
                                    books["large_thumbnail"])

#### VECTOR DATABASE
raw_documents = TextLoader("tagged_description.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=0.1, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)
db_books= Chroma.from_documents(documents, embedding=OpenAIEmbeddings())

#### Retrieve semantic recs from data set and apply filtering based on category and sorting by emotional tone
def get_semantic_recommendations(query: str,
                                 category: str = None,
                                 tone: str = None,
                                 initial_top_k = 50,
                                 final_top_k = 16) -> pd.DataFrame:
    # get recommendations
    recs = db_books.similarity_search(query, k=initial_top_k)

    # get back isbn of rec by splitting off
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]

    # get only the books that were recommended
    books_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)


    # apply filtering based on category (through dropdown)
    if category != "All":
        books_recs = books_recs[books_recs["simple_categories"] == category].head(final_top_k)
    else: # list all the books within limit
        books_recs = books_recs.head(final_top_k)

    # sort based on probability of emotions for tone of books -> left out neutral and disgust
    if tone == "Happy":
        books_recs.sort_values(by=["joy"], ascending=False, inplace=True)
    elif tone == "Surprising":
        books_recs.sort_values(by=["surprise"], ascending=False, inplace=True)
    elif tone == "Angry":
        books_recs.sort_values(by=["anger"], ascending=False, inplace=True)
    elif tone == "Suspenseful":
        books_recs.sort_values(by=["fear"], ascending=False, inplace=True)
    elif tone == "Sad":
        books_recs.sort_values(by=["sadness"], ascending=False, inplace=True)


    return books_recs


#### create function to specify what we want to specify on the gradio dashboard
def recommend_books(query: str,
                    category: str,
                    tone: str):

    recommendations = get_semantic_recommendations(query, category, tone) # get the recommendations
    results = []

    # description: display description but only a certain amount of words
    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..." # attach an ellipses

        # authors: if books have multiple authors, combine using semicolon
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        # display info about book as a caption at bottom of book thumbnail
        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results


#### create the dashboard

# start with two lists
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    # title
    gr.Markdown("# Semantic Book Recommender")

    # user interactions
    with gr.Row():
        user_query = gr.Textbox(label="Please enter a description of a book:",
                                placeholder="e.g., A story about a thrilling adventure and pirates")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value  ="All")
        tone_dropdown = gr.Dropdown(choices = tones, label="Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations!")

    # show results as a gallery
    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended Books", columns= 8, rows = 2)

    # when user hits submit button
    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)

if __name__ == "__main__":
    dashboard.launch()
