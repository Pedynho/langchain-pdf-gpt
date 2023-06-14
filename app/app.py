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
        chunk_size=2000,
        chunk_overlap=0,
        length_function=len
      )
      chunks = text_splitter.split_text(text)

      encoding = tiktoken.encoding_for_model('gpt-3.5-turbo-16k')
      value = chunks[0] + chunks[1] + chunks[2] + chunks[3]
      print(value)
      print(len(encoding.encode(value)))

      # create embeddings
      model_name = "text-embedding-ada-002"
      embeddings = OpenAIEmbeddings(model=model_name)
      try:
        knowledge_base = FAISS.load_local("PDFVectorBase", embeddings)
      except RuntimeError as e:
        st.write('Num tá tendo não')
      
      # st.write('continua depois do try')
      # knowledge_base = FAISS.from_texts(chunks, embeddings)
      # knowledge_base.save_local('PDFVectorBase')
      # print(knowledge_base)
      # #persisting the knowledge_base to futher verifications
      # FAISS.write_index(knowledge_base.index, 'knowledge_base.index')
      # # Trying to load the knowledge_base
      # knowledge_base = FAISS.read_index('knowledge_base.index')
      # print(knowledge_base)

      # # show user input
      # question = st.text_input("Ask a question about your PDF:")

      # if question:
        
      #   # busca bloco de texto que melhor responde a pergunta
      #   docs = knowledge_base.similarity_search(question)
      #   print(docs)
      #   #termos a serem inseridos no dissertação
      #   termos = "['Empírico', 'Demonstração', 'Iluminismo', 'Novo mundo', 'Antropologia', 'Ciência', 'Cultura', 'Alteridade', 'Método indutivo', 'Século XVIII', 'Fatos sociais', 'Selvagem', 'Civilizado', 'Religião', 'Filosofia']"
      #   # prompt template
      #   template = """
      #   construa um texto dissertativo-argumentativo com base no conteúdo a seguir respondendo a seguinte pergunta: {question}
      #   ---------------------
      #   {doc}
      #   ---------------------
      #   siga a seguinte estrutura: <[introdução][desenvolvimento][conclusão]> 
      #   adicione alguns do seguintes termos: {termos}
      #   """
      #   prompt = PromptTemplate(
      #     input_variables=['doc', 'question', 'termos'],
      #     template=template
      #   )
      #   # llminput = prompt.format(doc=docs[0]['page_content'], question=question, termos=termos)
      #   # print(llminput, prompt)
      #   # st.write(llminput)
        
      #   # llm = OpenAI()
      #   # chain = LLMChain(llm, prompt=prompt)
      #   # with get_openai_callback() as cb:
      #   #   response = chain.run(input_documents=docs, question=user_question)
      #   #   print(cb)
       
      #   # st.write(response)
    

if __name__ == '__main__':
    main()
