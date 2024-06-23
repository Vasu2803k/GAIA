# Environment variables
from dotenv import load_dotenv
load_dotenv()

# langchain_community for document loaders and vectorstores
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains.summarize import load_summarize_chain

from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from langchain_core.prompts import PromptTemplate

from langchain_community.vectorstores.faiss import FAISS
from langchain.globals import set_verbose, get_verbose

set_verbose(True)

import os
import time
from datetime import datetime
import concurrent.futures

import warnings
warnings.filterwarnings("ignore")
warnings.resetwarnings()


class text_augmentation():
    def __init__(self):
        self.model=ChatOpenAI(temperature=0.1, model='gpt-4-turbo-preview',model_kwargs={"top_p":0.9, "presence_penalty":0.6}, max_tokens=4096)
        self.embeddings_model=OpenAIEmbeddings(model="text-embedding-3-large", dimensions=2048)
        
    # Load the text files
    def document_loader(self,text_file):
        loader=TextLoader(text_file)
        loaded_text=loader.load()
        return loaded_text
    
    def get_document_chunks(self,text):
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n\n", "\n\n", "\n", ",", " "],
                chunk_size=5000,
                chunk_overlap=500,
                is_separator_regex=False
            )
            chunks = text_splitter.split_documents(text)
            return chunks
        except Exception as e:
            print("An error occurred:", e)
            return None

    def get_all_document_chunks(self,root_dir):
        text_files=[]
        for root, dirs, files in os.walk(root_dir):
            if(len(files)>0):
                text_files.extend([os.path.join(root,file) for file in files if os.path.splitext(os.path.basename(file))[1].lower()=='.txt'])

        doc_chunks=[]
        for text_file in text_files:
            doc_chunks.extend(self.get_document_chunks(self.document_loader(text_file)))
        return doc_chunks
    
    # Create a vectorstore
    def get_vectorstore(self,doc_chunks):
        persistent_dir="GAIA_GPT/faiss_index_summary"
        if not os.path.exists(persistent_dir):
            os.makedirs(persistent_dir)
        vectorstore=FAISS.from_documents(documents=doc_chunks, embedding=self.embeddings_model)
        vectorstore.save_local(persistent_dir)
        return vectorstore
    
    def summarisation(self,split_docs, doc_type):  

        if doc_type=="10k":
            prompt_template = """Write a concise summary of the following Freeport-McRonan 10-K report delimited by triple backticks with a focus on:
            1. Business operations: Provide a detailed summary of the company's core products, services, and revenue generation strategies.
            Highlight key operational facts with supporting evidence.
            2. Risk factors: Enumerate and analyze the top risks facing the company, arranging them by their potential impact on future operations. Support each risk factor with factual details from the report.
            3. Selected financial data: Examine the financial performance trends over the past five years. Include summarized tables that capture crucial financial metrics, noting any significant changes or patterns.
            4. MD&A insights: Analyze management's discussion and analysis section to extract insights on the company's past fiscal year performance. Summarize the management’s perspective on financial conditions and operational results,
            providing evidence from the narrative.
            5. Financial statements review: Summarize key elements from the income statement, balance sheets, and statement of cash flows. Highlight any critical observations made by the independent auditor in their letter, including summarized tables for essential financial data.

            Organize the findings in a logical manner that enhances clarity and understanding, ensuring each section is backed by relevant tables and factual evidence from the report to illustrate Freeport McMoRan's financial health and strategic direction

            ```{text}```

            CONCISE SUMMARY:"""
            
            prompt = PromptTemplate.from_template(prompt_template)
        elif doc_type=="annual reports":
            prompt_template = """Summarize the key components of the annual report delimited by triple backticks, providing a comprehensive overview of the company's activities and financial performance over the last year, with an emphasis on: 
            1. Letter to shareholders: Extract key messages and themes from the letter, highlighting leadership's perspective on the company’s performance and future outlook.
            2. Business and industry overview: Summarize the company's position within its industry, including any significant changes or developments in its business model or competitive landscape.
            3. Performance highlights: Identify and outline the major achievements and challenges faced by the company over the past year, supported by relevant data and metrics.
            4. Audited financial statements: Provide a concise analysis of the balance sheet, income statement, and statement of cash flows, focusing on key financial indicators and trends. Include simplified tables or graphics to represent crucial data points effectively.
            5. Notes to the financial statements: Highlight critical notes that shed light on the numbers, such as accounting policies, commitments, and contingencies.
            6. Future outlook: Summarize the company’s performance projections and strategic plans for the upcoming years, based on the discussion in the report.

            Organize the findings in a logical manner that enhances clarity and understanding, ensuring each section is backed by relevant tables and factual evidence from the report.
        
            ```{text}```

            CONCISE SUMMARY:"""
            prompt = PromptTemplate.from_template(prompt_template)

        elif doc_type=="general meeting notice":
            prompt_template = """Create a concise summary for the upcoming shareholders' meeting notice, ensuring it effectively communicates the necessary details to inform and prepare stakeholders for the meeting.
            The summary should cover:
            1. Meeting logistics: Clearly state the time, date, and location of the meeting, including any options for remote participation.
            2. Purpose of the meeting: Outline the main objectives and agenda items, highlighting the key issues and decisions that will be addressed.
            3. Financial performance review: Summarize the company's financial health and performance over the relevant period, focusing on essential metrics and achievements. Reference specific documents or sections where detailed financial data and analyses can be found.
            4. Audit reports: Provide an overview of the latest audit findings, emphasizing any significant matters that require shareholder attention or action. Mention where the full audit report is accessible for detailed review.
            5. Legal matters: Briefly describe any current legal issues or proceedings affecting the company, including potential impacts on the company’s operations or financial status. Indicate where more detailed information is available.
            6. Decision-making information: Highlight critical information or documents that shareholders should review before the meeting to participate effectively in decision-making processes.

            Ensure the summary is structured to offer stakeholders a clear and comprehensive overview of what to expect facilitating informed participation and decision-making at the meeting

            ```{text}```

            CONCISE SUMMARY:"""
            prompt = PromptTemplate.from_template(prompt_template)
        elif doc_type=="sustainability":
            prompt_template="""Summarize the Freeport-McMoRan sustainability report delimited by triple backticks, focusing on the following key areas:

            1. Environmental Stewardship: Detail the company's efforts in managing their environmental impact. Highlight any new initiatives or technologies adopted in the reporting period.
            2. Social Responsibility: Describe Freeport-McMoRan's initiatives regarding community engagement, employee welfare, and social investments. Include information on health and safety measures, community development programs, and any partnerships with local or global organizations.
            3. Governance: Provide insights into the company's governance practices, including ethical business conduct, compliance standards, and transparent reporting mechanisms. Note any changes or improvements in governance structures or policies.
            4. Economic Performance: Summarize the company’s economic contributions, including direct and indirect economic impacts, investments in sustainability, and financial performance related to sustainable practices.
            5. Comparative Analysis: Compare the current report's findings and achievements with those of the previous years. Highlight any trends, improvements, or areas requiring further attention.

            Insights and Future Outlook: Conclude with an analysis of the report, offering insights into the company's sustainability trajectory. Discuss any future goals, targets, or commitments made by Freeport-McMoRan towards enhancing its sustainability efforts.

            ```{text}```

            Please ensure the summary is concise, informative, and provides a balanced view of Freeport-McMoRan's sustainability performance and strategic direction.
            """
        refine_template="""
        Your task is to review the given summary and the additional context provided in triple backticks. If the new information does not substantially alter or enhance the understanding of the subject, return the original summary without preamble or modification. If the new context significantly changes or enriches the original summary, revise it accordingly. Focus solely on integrating crucial new details into the existing summary if needed, without prefacing your response with evaluations of the original summary's comprehensiveness or quality.
        
        Given summary: ```{existing_answer}```

        Additional context: "------------\n"
            "{text}\n"
        "------------\n"
        Based on the above, directly integrate any vital new insights into the original summary if necessary. Otherwise, return the original summary as is without providing the additional answer.
        """
        refine_prompt = PromptTemplate.from_template(refine_template)
        chain = load_summarize_chain(
            llm=self.model,
            chain_type="refine",
            question_prompt=prompt,
            refine_prompt=refine_prompt,
            input_key="input_documents",
            output_key="output_text",
        )
        result = chain.invoke({"input_documents": split_docs}, return_only_outputs=True)

        return result["output_text"]
        
    def text_file_overwrite(self, file_path, doc_type):
        
        output_dir="GAIA_GPT/Text_Files1_overwritten"
        # Extract folder name as the file name
        folder_name = os.path.basename(os.path.dirname(file_path))

        # Construct output file path
        # output_file for pdf's
        folder_name_=file_path.split('/')[-3]
        output_file = os.path.join(output_dir, folder_name_, folder_name, f"{os.path.splitext(os.path.basename(file_path))[0]}.txt")

        # Create the directory for the output file if it doesn't exist - check by exist_ok parameter
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        loaded_text=self.document_loader(file_path)

        
        doc_chunks=self.get_document_chunks(loaded_text)

        summarised_text=self.summarisation(doc_chunks, doc_type)

        # Write extracted text to the output file
        with open(output_file, "a") as f:
            f.write(summarised_text)
            f.write(loaded_text[0].page_content)

        print(f"summarised the file from '{file_path}' and saved as '{output_file}'")
        

    def process_text_files(self,root_dir, doc_type):

        text_files = []

        # Collect all text files
        for root, dirs, files in os.walk(root_dir):
            # If there are any files within the directory
            if len(files)>0:
                text_files.extend([os.path.join(root, file) for file in files if file.endswith(".txt")])
        
        # Process PDF files using multithreading
        #with concurrent.futures.ThreadPoolExecutor() as executor:
            #executor.map(self.text_file_overwrite, text_files)
        for text_file in text_files:
            self.text_file_overwrite(text_file, doc_type)

if __name__=="__main__":
    output_dirs=["/home/terradxllm/GAIA/GAIA_GPT/Text_Files1/Annual_Proxy_PDFs/10-K",
                     "/home/terradxllm/GAIA/GAIA_GPT/Text_Files1/Annual_Proxy_PDFs/Annual",
                     "/home/terradxllm/GAIA/GAIA_GPT/Text_Files1/Annual_Proxy_PDFs/Proxy",
                     "/home/terradxllm/GAIA/GAIA_GPT/Text_Files1/Sustainability_PDFs"]
        
    for i, output_dir in enumerate(output_dirs):

        textfiles_root_dir=output_dir

        text_augmentation_obj=text_augmentation()
        if i==0:
            doc_type="10k"
        elif i==1:
            doc_type="annual reports"
        elif i==2:
            doc_type="general meeting notice"
        elif i==3:
            doc_type="sustainability"

        # Process text files in the root directory using multithreading
        text_augmentation_obj.process_text_files(textfiles_root_dir, doc_type)

        text_files_dir_faiss_index="GAIA_GPT/Text_Files1_overwritten"

        user_query = input("Do you want to create new vectorstore database?...!\n")

        if user_query.strip().lower()=="yes":
            vectorstore=text_augmentation_obj.get_vectorstore(text_augmentation_obj.get_all_document_chunks(text_files_dir_faiss_index))
        else:
            exit()