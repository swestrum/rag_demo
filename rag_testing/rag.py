from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from fastembed.sparse.bm25 import Bm25
from fastembed.late_interaction import LateInteractionTextEmbedding
from textwrap import dedent
import json
import tqdm
import ollama


# Set up Qdrant client
client = QdrantClient(url="http://localhost:6333", timeout=10000)

# Set up some basic information
collection_name = "pentester_land"
dense_model_name = "BAAI/bge-small-en-v1.5"
sparse_model_name = "Qdrant/bm25"
late_interaction_model_name = "colbert-ir/colbertv2.0"

# Load the pentesterland json
pentesterland_json = "./pentester_land_8_15.json"

# Create dense embeddings
dense_embedding_model = TextEmbedding(dense_model_name)

# Create sparse embeddings
bm25_embedding_model = Bm25(sparse_model_name)

# Create late interaction embeddings
late_interaction_embedding_model = LateInteractionTextEmbedding(
    late_interaction_model_name
)


def batched(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def load_articles(client, input_file):
    articles = []
    with open(input_file, "r") as file:
        data = json.load(file)
    articles = data["data"]
    for i, article in enumerate(articles):
        article["text"] = json.dumps(article)
        article["id"] = i
    articles_str = [str(a) for a in articles]
    articles_str = articles_str[0]

    # Put embeddings in qdrant
    try:
        client.create_collection(
            collection_name,
            vectors_config={
                dense_model_name.split("/")[1]: models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE,
                ),
                # late_interaction_model_name.split("/")[1]: models.VectorParams(
                #     size=128,
                #     distance=models.Distance.COSINE,
                #     multivector_config=models.MultiVectorConfig(
                #         comparator=models.MultiVectorComparator.MAX_SIM,
                #     ),
                # ),
            },
            sparse_vectors_config={
                sparse_model_name.split("/")[1]: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )
    except:
        print("Pentester collection already exists, skipping data load")
        return False

    batch_size = 10
    for batch in tqdm.tqdm(
        batched(articles, batch_size), total=len(articles) // batch_size
    ):
        articles_text = [a["text"] for a in batch]
        dense_embeddings = list(dense_embedding_model.passage_embed(articles_text))
        bm25_embeddings = list(bm25_embedding_model.passage_embed(articles_text))
        # late_interaction_embeddings = list(
        #     late_interaction_embedding_model.passage_embed(articles_text)
        # )

        client.upload_points(
            collection_name,
            points=[
                models.PointStruct(
                    id=int(batch[i]["id"]),
                    vector={
                        dense_model_name.split("/")[1]: dense_embeddings[i].tolist(),
                        sparse_model_name.split("/")[1]: bm25_embeddings[i].as_object(),
                        # late_interaction_model_name.split("/")[
                        #     1
                        # ]: late_interaction_embeddings[i].tolist(),
                    },
                    payload={
                        "id": batch[i]["id"],
                        "bugs": batch[i]["Bugs"],
                        "programs": batch[i]["Programs"],
                        "text": batch[i]["text"],
                    },
                )
                for i, _ in enumerate(batch)
            ],
            # We send a lot of embeddings at once, so it's best to reduce the batch size.
            # Otherwise, we would have gigantic requests sent for each batch and we can
            # easily reach the maximum size of a single request.
            batch_size=batch_size,
        )
    return True


def query_rag(query):
    dense_query_vector = next(dense_embedding_model.query_embed(query))
    sparse_query_vector = next(bm25_embedding_model.query_embed(query))
    late_query_vector = next(late_interaction_embedding_model.query_embed(query))

    prefetch = [
        models.Prefetch(
            query=dense_query_vector,
            using=dense_model_name.split("/")[1],
            limit=20,
        ),
        models.Prefetch(
            query=models.SparseVector(**sparse_query_vector.as_object()),
            using=sparse_model_name.split("/")[1],
            limit=20,
        ),
        # models.Prefetch(
        #     query=late_query_vector,
        #     using=late_interaction_model_name.split("/")[1],
        #     limit=20,
        # ),
    ]
    results = client.query_points(
        collection_name,
        prefetch=prefetch,
        query=models.FusionQuery(
            fusion=models.Fusion.RRF,
        ),
        with_payload=False,
        limit=5,
    )

    points = {point.id: point.score for point in results.points}
    retrieved_points = client.retrieve(collection_name, ids=points.keys())
    titles = []
    for point in retrieved_points:
        data = json.loads(point.payload["text"])
        for link in data["Links"]:
            titles.append(link["Title"])
    print(f"\nFound {len(titles)} bug bounty writeups related to your request!")
    prompt_template = dedent(
        f"""
        Here are several bug bounty writeups you have looked up that are relevant to your task:

        - {{articles}}

        Given that information, answer the following question:

        {{query}}

        Remember to answer concicely, and be as helpful as possible. Qualify your statements, 
        because the provided article context does not represent all the information available 
        on this question.
        """
    )

    prompt = prompt_template.format(
        articles="- ".join([title + "\n" for title in titles]), query=query
    )
    response = ollama.chat(
        model="llama3.1",
        messages=[
            {
                "role": "sytem",
                "content": "You are a helpful assistant to a security researcher, whose primary task is to summarize relevant information to speed up penetration testing.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    return response["message"]["content"]


if __name__ == "__main__":
    load_articles(client=client, input_file=pentesterland_json)
    print("Enter your question, or /bye to quit")
    query = ""
    while True:
        query = input(">>> ")
        if query == "/bye":
            print("\nGoodbye, thanks for chatting!\n")
            break
        print(f"\nLLM: {query_rag(query)}\n")
