from dotenv import load_dotenv
import streamlit as st
import tiktoken
from PyPDF2 import PdfReader
from langchain import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
        
      # split into chunks
      text_splitter = CharacterTextSplitter(
        separator=".",
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len
      )
      chunks = text_splitter.split_text(text)

      # create embeddings
      model_name = "text-embedding-ada-002"
      embeddings = OpenAIEmbeddings(model=model_name)
      try:
        knowledge_base = FAISS.load_local("PDFVectorBase", embeddings)
      except RuntimeError as e:
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        knowledge_base.save_local('PDFVectorBase')

      # show user input
      question = st.text_input("Ask a question about your PDF:")

      if question:
        
        # busca bloco de texto que melhor responde a pergunta
        docs = knowledge_base.similarity_search_with_relevance_scores(question,k=32)
        context = ''
        for doc in docs:
           context += " | " + doc[0].page_content

        # prompt template
        template = """
          ---------------------
          {context}
          ---------------------
          com base no conteúdo acima construa um texto dissertativo-argumentativo respondendo a seguinte pergunta: {question}
          siga a seguinte estrutura: introdução, desenvolvimento, conclusão 
          adicione alguns do seguintes termos: Empírico, Demonstração, Iluminismo, Novo mundo, Antropologia, Ciência, Cultura, Alteridade, Método indutivo, Século XVIII, Fatos sociais, Selvagem, Civilizado, Religião, Filosofia.
        """
        prompt = PromptTemplate(
          input_variables=['context', 'question'],
          template=template
        )

        #  medidindo tamnho do input em tokens
        llminput = prompt.format(context=context, question=question)
        encoding = tiktoken.encoding_for_model('gpt-3.5-turbo-16k')
        print(len(encoding.encode(llminput)))
        
        # # enviado pompt para o LLM
        llm = OpenAI(model_name="gpt-3.5-turbo-16k")
        chain = LLMChain(llm=llm, prompt=prompt)

        with get_openai_callback() as cb:
          response = chain.run(context=context, question=question)
          print(cb)
       
        st.write(response)
    

if __name__ == '__main__':
    main()
