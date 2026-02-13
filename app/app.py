# app/app.py
import streamlit as st
import keras_nlp
import keras
import os

# Set page config
st.set_page_config(page_title="Math Assistant", page_icon="🧮")
st.title("🧮 Math Assistant (Fine-tuned Gemma)")
st.markdown("Ask any mathematics question and get an answer from the fine-tuned model.")

@st.cache_resource
def load_model():
    """Load the fine-tuned model from the saved directory."""
    model_path = os.path.join(os.path.dirname(__file__), "..", "model")
    if not os.path.exists(model_path):
        st.error("Model not found. Please train the model first.")
        return None, None
    # Load tokenizer and model
    tokenizer = keras_nlp.models.GemmaTokenizer.from_preset(model_path)
    model = keras_nlp.models.GemmaCausalLM.from_preset(model_path)
    return model, tokenizer

model, tokenizer = load_model()

if model is None or tokenizer is None:
    st.stop()

# Input from user
question = st.text_area("Enter your math question:", height=100, placeholder="e.g., Solve for x: 2x + 5 = 13")

if st.button("Generate Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            # Format prompt
            prompt = f"Question: {question}\nAnswer:"
            # Generate
            output = model.generate(prompt, max_length=512)
            # Extract answer (remove the prompt part)
            answer = output[len(prompt):].strip()
            st.subheader("Answer:")
            st.write(answer)

# Sidebar with examples
st.sidebar.header("Sample Questions")
examples = [
    "What is the derivative of x^2?",
    "Solve the quadratic equation x^2 - 5x + 6 = 0",
    "If a circle has radius 5, what is its area?",
    "Compute the limit of (sin x)/x as x approaches 0",
]
for ex in examples:
    if st.sidebar.button(ex):
        question = ex  # Would need to set session state, but we'll just show
        # For simplicity, we'll just show the question in the main area; user must click generate
        st.session_state["question"] = ex
        # Rerun? Not needed; we can just set text area value.
        # We'll use st.experimental_rerun? Better to use session state.
        pass

# If session state has a question, pre-fill the text area
if "question" in st.session_state:
    st.text_area("Enter your math question:", value=st.session_state["question"], key="input_question")
else:
    st.text_area("Enter your math question:", key="input_question")