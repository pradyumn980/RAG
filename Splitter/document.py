from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

splitter = RecursiveCharacterTextSplitter.from_language(
    chunk_size=20,  
    chunk_overlap=0,
    language=Language.JAVA,
)

text='''class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}'''

result = splitter.split_text(text)
print(result[1])