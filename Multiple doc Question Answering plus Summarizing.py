#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install --upgrade langchain openai -q')
get_ipython().system('pip install unstructured -q')
get_ipython().system('pip install unstructured[local-inference] -q')
get_ipython().system('pip install detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.6#egg=detectron2 -q')
get_ipython().system('apt-get install poppler-utils')
get_ipython().system('pip install tiktoken -q')
get_ipython().system('pip install chromadb -q')
get_ipython().system('pip install pinecone-client -q')


# In[ ]:


import os
os.environ["OPENAI_API_KEY"] = " API KEY HERE "


# In[ ]:


import os
import openai
import pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
from langchain.chains.summarize import  load_summarize_chain
from IPython.display import Markdown, display


# In[ ]:


# Loading multiple documents here with different subjects, to measure the acuuracy of the final output

files = '/PATH TO THE FILES'
fs = os.path.join(files)
loader = DirectoryLoader(fs)
documents = loader.load()
print(len(documents))


# In[ ]:


def split_docs(documents, chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", "(?<=\. )", " ", ""]):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators = separators)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
print(len(docs))


# In[ ]:


embeddings = OpenAIEmbeddings()

query_result = embeddings.embed_query("Hello world")
len(query_result)


# In[ ]:


pinecone.init(
    api_key=" pinecone api key here",
    environment=" different from case to case "
)

index_name = "langchain-demo"

index = Pinecone.from_documents(docs,
                                embeddings, index_name=index_name,
                                #persist_directory = persist_directory
                                )


# In[ ]:


def get_similiar_docs(query, k=2, score=False):
  if score:
    similar_docs = index.similarity_search_with_score(query, k=k)
  else:
    similar_docs = index.similarity_search(query, k=k)
  return similar_docs


# In[ ]:


#model_name = "text-davinci-003"
# model_name = "gpt-3.5-turbo"
model_name = "gpt-4"
llm = OpenAI(model_name=model_name,temperature = 0.3)



#memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = load_qa_chain(llm, chain_type="stuff")
#qa = ConversationalRetrievalChain.from_llm(llm , index.as_retriever(), memory=memory)

def get_answer(query):
  similar_docs = get_similiar_docs(query)
  answer = chain.run(input_documents=similar_docs, question=query)
  #answer = qa({"question": query})
  return answer


# In[ ]:


query = " WRITE DOWN YOUR QUESTION HERE"
answer = get_answer(query)


# Summarizing the documents with LangChain :

# In[ ]:


# for summarizing each part
chain.llm_chain.prompt.template


# In[ ]:


# for combining the parts
chain.combine_document_chain.llm_chain.prompt.template


# In[ ]:


chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)


output_summary = chain.run(docs)
wrapped_text = textwrap.fill(output_summary, width=100, break_long_words=False, replace_whitespace=False)


# In[ ]:




