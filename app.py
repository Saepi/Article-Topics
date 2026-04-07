import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json

MODEL_NAME = "checkpoint"
TEMPERATURE = 0.18
with open("idx2label.json", "r") as file:
    IDX2LABEL = json.load(file)

BASE_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(IDX2LABEL)
    )

    classifier_state = torch.load("classifier_head.pt", map_location="cpu")
    model.classifier.load_state_dict(classifier_state)

    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

def predict(title, abstract, temperature=TEMPERATURE):
    text = f"{title} {abstract}"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits / temperature, dim=-1)

    return probs.squeeze().tolist()


st.title("Article Category Classifier")

st.markdown("Type the article title and abstract to obtain the article's categories.")

title_input = st.text_input("Title")
abstract_input = st.text_area("Abstract", height=200)

if st.button("Determine categories"):
    if not title_input.strip():
        st.error("Please, add title")
    elif not abstract_input.strip():
        st.error("Please, add abstract")
    else:
        try:
            with st.spinner("Model works..."):
                probs = predict(title_input, abstract_input)

            results = sorted(
                zip(range(len(probs)), probs),
                key=lambda x: x[1],
                reverse=True
            )

            st.success("Ready!")

            total_prob = 0
            for label, score in results:
                if total_prob < 0.95:
                    category = IDX2LABEL[str(label)]
                    st.write(f"**{category}** — {score:.2%}")
                    total_prob += score
                else:
                    break

        except Exception as e:
            st.error("There was some error during the work. Try again")
            st.exception(e)