from backend.ai.llm import get_llm


DOMAIN_CONTEXT = """
You are the query-rewriting component of IP-SAKTI Sahayak.

IP-SAKTI Sahayak is an AI assistant focused on:
- Ayurveda
- Indian traditional knowledge
- Intellectual Property Rights (IPR)
- Patents
- Prior art
- TKDL
- Access and Benefit Sharing (ABS)
- Biodiversity
- Trademarks
- Designs
- Copyright
- Trade secrets
- AYUSH and product classification
- Regulatory requirements
- International intellectual property frameworks

Your task is to rewrite the user's query into ONE complete,
clear and retrieval-friendly question.

The rewritten query will be used for:
1. Query classification
2. Hybrid semantic/keyword retrieval
3. CrossEncoder reranking

Important rules:

1. Preserve the user's original intent.
2. Do not answer the question.
3. Do not add facts that are not reasonably implied by the query.
4. Expand common abbreviations when their meaning is clear.
5. Convert incomplete queries into complete questions.
6. Convert conversational language into precise search language.
7. Preserve important legal/IP terminology.
8. If the user uses "IPR", expand it to "Intellectual Property Rights".
9. If the user uses "TKDL", expand it to "Traditional Knowledge Digital Library (TKDL)".
10. If the user uses "ABS", expand it to "Access and Benefit Sharing (ABS)".
11. If the query is already clear and complete, return a cleaned-up version.
12. Keep the query concise.
13. Do not introduce a specific law, section, country, product type,
    or legal conclusion unless the user clearly indicated it.
14. For a very short domain query such as "IPR", "patent",
    "TKDL", or "ABS", convert it into a natural complete question
    appropriate for the IP-SAKTI domain.
15. Return ONLY the rewritten query.
"""


def rewrite_query(query, language="en"):

    if not query:
        return query

    query = query.strip()

    if not query:
        return query

    # Very short queries benefit most from rewriting.
    # Longer, already-clear queries are also cleaned by the LLM.
    prompt = f"""
{DOMAIN_CONTEXT}

User language: {language}

Original user query:
{query}

Rewrite the query now.

Return ONLY the rewritten query.
"""

    try:

        llm = get_llm()

        response = llm.invoke(prompt)

        rewritten = response.content.strip()

        # Remove accidental quotation marks
        rewritten = rewritten.strip(
            "\"'`"
        )

        if not rewritten:
            return query

        return rewritten

    except Exception as e:

        print(
            f"QUERY REWRITER ERROR: {e}"
        )

        # Never break the RAG pipeline just because
        # rewriting failed.
        return query