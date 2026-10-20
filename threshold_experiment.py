from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

knowledge_base = [
    # Python Development Topic
    ("python", "Python uses indentation to define code blocks instead of curly braces."),
    ("python", "A virtual environment keeps a project's packages separate from other projects."),
    ("python", "FastAPI lets you build web APIs quickly with automatic documentation."),
    ("python", "Pytest makes it easy to write and run unit tests for your functions."),
    # Cooking Topic
    ("cooking", "Searing meat at high heat creates a flavorful brown crust."),
    ("cooking", "Bread dough needs time to rise so the yeast can produce gas."),
    ("cooking", "Adding salt to pasta water seasons the noodles as they cook."),
    ("cooking", "A sharp knife is safer than a dull one because it slips less."),
    # Space Topic
    ("space", "Mars is called the red planet because of iron oxide on its surface."),
    ("space", "A light year measures the distance light travels in one year."),
    ("space", "The International Space Station orbits Earth about every 90 minutes."),
    (
        "space",
        "Elon Musk is the founder of SpaceX, a company that develops and launches advanced rockets and spacecraft.",
    ),
    # Music Topic
    (
        "music",
        "A flute is a long metal or wooden wind instrument that is played by blowing air across the mouthpiece.",
    ),
    ("music", "Tempo describes how fast or slow a piece of music is played."),
    ("music", "A guitar is a 6-stringed instrument with a hollow body."),
    ("music", "Practicing scales helps build finger speed and accuracy on the piano."),
]

test_queries = [
    # Clear matches
    ("How do I test my Python code?", ["python"]),
    ("What planet is red?", ["space"]),
    ("How do I make bread at home?", ["cooking"]),
    # Ambiguous
    ("How do I get faster with practice?", ["music", "cooking"]),
    ("How do I launch something quickly?", ["space", "python"]),
]

# Encode the knowledge base once, before looping through the queries
doc_sentences = [sentence for topic, sentence in knowledge_base]
doc_embeddings = model.encode(doc_sentences, convert_to_tensor=True)

THRESHOLDS = [0.3, 0.5, 0.7]

for query, relevant_topics in test_queries:
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, doc_embeddings)[0]

    print("=" * 50)
    print(f'Query: "{query}"')
    print(f"Relevant Topic(s): {', '.join(relevant_topics)}")

    for threshold in THRESHOLDS:
        passed = []
        missed = []

        for i, score in enumerate(scores):
            topic, sentence = knowledge_base[i]
            score = score.item()
            is_relevant = topic in relevant_topics

            if score >= threshold:
                passed.append((score, topic, sentence, is_relevant))
            elif is_relevant:
                missed.append((score, topic, sentence))

        passed.sort(reverse=True)
        missed.sort(reverse=True)

        # How many results pass, and which ones
        result_word = "result" if len(passed) == 1 else "results"
        print(f"\n  Threshold {threshold}: {len(passed)} {result_word}")
        for score, topic, sentence, is_relevant in passed:
            label = "" if is_relevant else "  <-- off-topic"
            print(f"    [{score:.2f}] {sentence}{label}")

        # Which relevant results were missed at this threshold
        if missed:
            print(f"  Relevant results missed: {len(missed)}")
            for score, topic, sentence in missed:
                print(f"    [{score:.2f}] ({topic}) {sentence}")
        else:
            print("  No relevant results missed.")

    print()

print("=" * 50)