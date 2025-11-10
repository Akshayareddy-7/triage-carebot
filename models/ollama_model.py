# from ollama import Ollama

# class OllamaModel:
#     def __init__(self, model_name="your-model-name"):
#         self.ollama = Ollama()
#         self.model_name = model_name

#     def summarize(self, prompt: str) -> str:
#         """
#         Sends prompt to Ollama model and returns raw text output.
#         """
#         response = self.ollama.chat(
#             model=self.model_name,
#             messages=[{"role": "user", "content": prompt}]
#         )
#         return response["choices"][0]["message"]["content"]
