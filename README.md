# Exploring Vertex AI Search for Personalized Conversational RAGs: A POC Investigation

## Introduction

Vertex AI Search & Conversation is a serverless product capable of indexing
unstructured and structured data. While Langchain offers it as a retriever, this
Proof of Concept (POC) aimed to assess its suitability for per-user, persistent
conversations. This required storing user history alongside uploaded documents.

## Implementation and Findings

Firestore proved reliable for storing user history. However, document storage in
GCS and indexing with Vertex AI Search resulted in a disjointed workflow.
Tracking document locations across both systems felt cumbersome, requiring a
"pseudo-transactional" approach.

But the main issue was the lack of granular purge functionality in Vertex AI
Search, as it only supports wildcard filters. This limitation hinders effective
management of per-user conversational data.

## Conclusion

While Vertex AI Search demonstrates promise for knowledge base applications, its
current limitations make it less suitable for implementing personalized,
Response-Activation-Generation (RAG) systems at the individual user level.
