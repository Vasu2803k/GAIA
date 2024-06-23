# Environment variables
from dotenv import load_dotenv
load_dotenv()

import asyncio

# pathlib
from pathlib import Path

# llama-index
from llama_index.core import ServiceContext, VectorStoreIndex, SimpleDirectoryReader,  node_parser, StorageContext, load_index_from_storage

#from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import MessageRole, ChatMessage

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.llms.openai import OpenAI

from llama_index.vector_stores.faiss import FaissVectorStore

import os
import time
from datetime import datetime

import concurrent.futures
import warnings

warnings.filterwarnings("ignore")
import faiss

now = datetime.now()
# Format the date and time as a string
current_time_date_str = now.strftime("%Y-%m-%d %H:%M:%S")

class GAIA_GPT_llama_index:
    # Intialize openai environment
    def __init__(self):
        self.chat_history_llama_index=[ChatMessage(role=MessageRole.ASSISTANT, content="Hello there! I'm your assistant! Whether you're seeking information about Freeport McMoRan's operations, exploring annual reports or sustainaibility reports, I'm here to provide you with accurate and insightful answers. Let's dig into the fascinating world of geology and mining together!")]
        self.model=OpenAI(temperature=0.2, model='gpt-4-turbo-preview',model_kwargs={"top_p":0.9, "presence_penalty":0.6}, max_tokens=1024, streaming=True)

        self.embedding_model=OpenAIEmbedding(model="text-embedding-3-large", dimensions=1024)

        self.GAIA_RAG_llama_index()

    def get_llama_index(self, dir):
        documents=self.get_documents_llama_index(dir)
        #print(documents[0])
        # Initialize the parser
        parser=node_parser.SentenceSplitter.from_defaults(chunk_size=1024, chunk_overlap=128)
        # Parse documents into nodes
        nodes = parser.get_nodes_from_documents(documents)
        # Create and persist index
        index = VectorStoreIndex(nodes)

        index.storage_context.persist(persist_dir="/home/terradxllm/GAIA/GAIA_GPT/GAIA_Index")

        return index
    
    def get_documents_llama_index(self, root_dir):
        dir_paths = []
        for dir_path, sub_dir,  files in os.walk(root_dir):
            if files:
                dir_paths.append(dir_path)
        documents=[]
        for dir_path in dir_paths:
            documents.extend(SimpleDirectoryReader(dir_path).load_data())
        return documents
    
    def get_llama_index1(self, dir):
        documents=self.get_documents_llama_index(dir)
        #documents=SimpleDirectoryReader(dir).load_data()
        parser=node_parser.SentenceSplitter.from_defaults(chunk_size=1024, chunk_overlap=128, include_metadata=True)

        nodes=parser.get_nodes_from_documents(documents=documents)

        print(nodes[0].get_content)
        service_context=ServiceContext.from_defaults(llm=self.model,embed_model=self.embedding_model, chunk_size=1024, chunk_overlap=128)

    
        d=1024
        faiss_index = faiss.IndexFlatL2(d)
        # faiss storage
        vectorstore=FaissVectorStore(faiss_index=faiss_index)
        storage_context=StorageContext.from_defaults(vector_store=vectorstore)
        index=VectorStoreIndex(nodes, service_context=service_context, storage_context=storage_context)
        
        index.storage_context.persist(persist_dir="/home/terradxllm/GAIA/GAIA_GPT/GAIA_Index")

        return index
    def get_relevant_chat_history(self, num=3): # load only 3 last query and response chat betw ai and user
        if len(self.chat_history)>num:
            relavant_chat_history=self.chat_history[-num:]
        else:
            relavant_chat_history=self.chat_history
        return relavant_chat_history
    
    def get_response_llama_index(self, user_query, retriever):
        #memory = ChatMemoryBuffer.from_defaults(token_limit=20000)
        system_prompt=("""
            You will be provided with a context delimited by triple quotes. Your task is to analyse the context, Creating a table to organize data or results if the problem involves multiple data points or comparisons and assist the user in providing factual answers. You must follow certain instructions given below: 
            Instructions:
            - Create a table to organize data or results if the problem involves multiple data points or comparisons. Ensure the table is sorted in ascending order.
            - Break down the problem into smaller, manageable steps.
            - For each step, apply mathematical principles or calculations as required.
            - After completing the calculations, review each step and the final answer for accuracy.
            - Use clear and concise language to explain the reasoning behind each step and the significance of the final answer.
            Example:
            "Calculate the total cost of 5 items priced at $3.50, $4.75, $2.00, $8.25, and $6.50, including a 5% sales tax."
            
            Steps:
            1. Create and sort the table of item prices.
            2. Calculate the sum of the item prices.
            3. Apply the sales tax rate.
            4. Compute the total cost.
            5. Recheck calculations for accuracy.
             
            Final Answer: "The total cost of the items, including sales tax, is $26.25."
            Reasoning: "The total was calculated by summing the prices of all items and adding a 5% sales tax. Organizing data in a table ensured accuracy and clarity throughout the process."

            Notes:
            - Ensure accuracy by following each step methodically and rechecking calculations.
            - Utilize tables for better visualization and organization of data, aiding in a clearer calculation process."
            
            Given the context information and not prior knowledge, Answer the user question"""
            )

        chat_engine = ContextChatEngine.from_defaults(
            retriever=retriever,
            chat_history=self.chat_history_llama_index,
            llm=self.model,
            system_prompt=system_prompt,
            #user_prompt=user_prompt_template,
            verbose=True
        )
        streamed_response=chat_engine.stream_chat(user_query)
        response=""
        print("\n")
        for token in streamed_response.response_gen:
            response+=token
            print(token, end="", flush=True)
        print("\n")
        return response
    
    # Input query and response
    def GAIA_RAG_llama_index(self) -> None:
        user_query = input("Do you want to create new vectorstore database?...!\n")

        if user_query.strip().lower()=="yes":

            root_dir="/home/terradxllm/GAIA/GAIA_GPT/Text_Files"
            index=self.get_llama_index(root_dir)
            # Normal query engine
            #query_engine=index.as_query_engine(streaming=True, similarity_top_k=5)
    
        else:
            storage_context=StorageContext.from_defaults(persist_dir="/home/terradxllm/GAIA/GAIA_GPT/GAIA_Index")
            index=load_index_from_storage(storage_context=storage_context)
            
        retriever=index.as_retriever(similarity_top_k=50, verbose=True)
        
        print(self.chat_history_llama_index[0].content,"\n")
        # user input
        while True:
            user_query = input("Type your message here...!\n")
            if user_query.strip().lower()=="exit":
                break
            if user_query is not None and user_query != "":
                response=self.get_response_llama_index(user_query, retriever)
                self.chat_history_llama_index.append(ChatMessage(role=MessageRole.USER, content=user_query))
                self.chat_history_llama_index.append(ChatMessage(role=MessageRole.ASSISTANT, content=response))
        new_query=input("\n\nDo you need more help?\n")

        if(new_query.lower()=="yes"):

            self.chat_history_llama_index=[ChatMessage(role=MessageRole.ASSISTANT, content="Hello there! I'm your assistant! Whether you're seeking information about Freeport McMoRan's operations, exploring annual reports or sustainaibility reports, I'm here to provide you with accurate and insightful answers. Let's dig into the fascinating world of geology and mining together!")]

            self.GAIA_RAG_llama_index()
# question="How much capital expenditure did freeport have in 2004?"
# question="Provide the 1 page summary of the given Freeport?"
# question="Provide the breakup of the total capital expenditure in 2004"
if __name__=="__main__":
    GAIA_GPT_obj= GAIA_GPT_llama_index()
    