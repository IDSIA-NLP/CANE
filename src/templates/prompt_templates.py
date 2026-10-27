from langchain_core.prompts import PromptTemplate

ner_prompt = PromptTemplate.from_template("""
{context}

---
Input: {input}
Output:
""")