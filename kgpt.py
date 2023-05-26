from  PIL import Image
import os 
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI

# Name in the sidebar
st.set_page_config(page_title = 'Gyeongju GPT')
###################
def sidebar_bg():
   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url("https://cdn.pixabay.com/photo/2013/09/26/00/21/swan-186551_1280.jpg")
               }}
      </style>
      """,
      unsafe_allow_html=True,
      )
sidebar_bg()
###############################################
###############################################
# NAVIGATION BAR 
#https://discuss.streamlit.io/t/the-navigation-bar-im-trying-to-add-to-my-streamlit-app-is-blurred/24104/3
# First clean up the bar 
st.markdown(
"""
<style>
header[data-testid="stHeader"] {
    background: none;
}
</style>
""",
    unsafe_allow_html=True,
)
# Then put the followings (Data Prof: https://www.youtube.com/watch?v=hoPvOIJvrb8)
# Background color에 붉은빛이 약간 들어간 #ffeded
# https://color-hex.org/color/ffeded
st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)
st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #FFF7F7;">
  <a class="navbar-brand" href="https://digitalgovlab.com" target="_blank">Digital Governance Lab</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
    </ul>
  </div>
</nav>
<style>
    .navbar-brand{
    color: #89949F !important;
     }
    .nav-link disabled{
    color: #89949F !important;
     }
    .nav-link{
    color: #89949F !important;
     }
</style>
""", unsafe_allow_html=True)
#############################################################
#############################################################
##############################################################
#--- HIDE STREAMLIT STYLE ---
hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
#########################################################################
# LOGO
#https://pmbaumgartner.github.io/streamlitopedia/sizing-and-images.html
image = Image.open('logo_digitalgovlab_v2.jpg')
st.image(image, caption='')
########################
st.markdown(""" <style> .font2 {
     font-size:30px ; font-family: 'Cooper Black'; color: #000000;} 
     </style> """, unsafe_allow_html=True)
st.markdown('<p class="font2">경주 GPT (mini version)</p>', unsafe_allow_html=True) 
st.markdown(""" <style> .font3 {
     font-size:20px ; font-family: 'Cooper Black'; color: #000000;} 
     </style> """, unsafe_allow_html=True)
#st.markdown("##")
#st.markdown('<p class="font3">Wiki GPT 소개</p>', unsafe_allow_html=True) 
url = "https://platform.openai.com/account/api-keys"
st.markdown("""
경주 GPT는 기존 챗GPT에 경주시 관련 지식을 in-context learning 방식으로 추가 학습시킨 AI입니다.  \
경주 GPT는 챗GPT에 비해 경주에 관한 보다 더 상세한 정보를 제공할 수 있으며, \
한국정책학회 하계대회기간에 무료로 시연될 예정입니다. \
본 미니버젼은 랩의 예산 사정상 풀버젼(경주 GPT) 대비 약 10퍼센트 분량의 정보만을 습득하였습니다. \
본 미니버젼을 사용하기 위해서는 OpenAI에 유료계정과 API를 생성하셔야 합니다. [API Key 생성하러 가기](%s)
""" % url)
################################
#https://medium.com/@shashankvats/building-a-wikipedia-search-engine-with-langchain-and-streamlit-d63cb11181d0

global embeddings_flag
embeddings_flag = False

######################################
####################################################
#buff, col, buff2 = st.columns([1,3,1])
#st.markdown("##")
#https://pub.towardsai.net/building-a-q-a-bot-over-private-documents-with-openai-and-langchain-be975559c1e8
#chromadb 이슈는 아래 참조
#https://github.com/pypa/packaging-problems/issues/648
st.markdown("---")
st.markdown("OpenAI API Key를 입력해주세요 (API Key는 sk-로 시작합니다)")
openai_key = st.text_input(label=" ", label_visibility="collapsed")
os.environ["OPENAI_API_KEY"] = openai_key

if len(openai_key):
    model_id = "gpt-3.5-turbo"
    #model_id = "gpt-4"
    llm=ChatOpenAI(model_name = model_id, temperature=0.2)

    #loader1 = TextLoader('pdfdocs/test1.txt')
    #loader2 = TextLoader('pdfdocs/test2.txt')
    #mylist = [loader1, loader2]
    # Specify the directory path. Replace this with your own directory
    directory = './pdfdocs'
    # Get all files in the directory
    files = os.listdir(directory)
    # Create an emty list to store a list of all the files in the folder
    mylist = []
    # Iterate over the files
    for file_name in files:
        # Get the relative path of each file
        relative_path = os.path.join(directory, file_name)
        # Append the file to a list for further processing
        mylist.append(TextLoader(relative_path, encoding='utf8'))

    ########################################################
    # Save the documents from the list in to vector database 
    index = VectorstoreIndexCreator().from_loaders(mylist)

    st.markdown("---")
    st.markdown("신라의 천년고도 경주에 관하여 질문해주세요")
    prompt = st.text_input(label="XX", label_visibility="collapsed", key= "1")

    # Display the current response. No chat history is maintained
    if prompt:
        # stuff chain type sends all the relevant text chunks from the document to LLM    
        response = index.query(llm=llm, question = prompt, chain_type = 'stuff')

        # Write the results from the LLM to the UI
        st.write("<br><i>" + response + "</i><hr>", unsafe_allow_html=True )


#########

