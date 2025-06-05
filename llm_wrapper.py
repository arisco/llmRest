import os

class LLMWrapper:
    def __init__(self, mocked: bool = False):
        self.mocked = mocked
        if not mocked:
            from langchain_ollama import ChatOllama
            from langchain.schema import HumanMessage
            self.ChatOllama = ChatOllama
            self.HumanMessage = HumanMessage
            self.llm = ChatOllama(model="llama4:scout")
        else:
            self.llm = None

    def invoke(self, text: str, files_content: list[str] = None):
        if self.mocked:
            mock_response = "[MOCKED OLLAMA RESPONSE] Recibido: " + text
            if files_content:
                mock_response += "\n\n[Adjuntos]:\n" + "\n".join(files_content)
            return mock_response
        else:
            llm_input = text
            if files_content:
                for fc in files_content:
                    llm_input += f"\n\n[Adjunto]:\n{fc}"
            response = self.llm.invoke([self.HumanMessage(content=llm_input)])
            return response.content
