import os
from openai import OpenAI


def query(user_query: str, documents_text: list[str], documents_priorities: list[int], b64_image_urls: list[str], model_to_use: str='meta-llama/llama-4-scout'):
  '''
    Query an LLM given RAG context.

    Input:
      user_query: the actual prompt that the user is asking about
      documents_text: a list of documents retrieved with RAG
      documents_priorities: a list of priorities (integers) associated with each document
      b64_image_urls: if any documents retrieved were images, the Base64 encoded image urls
      model_to_use: which model to query (if images are used, must be a VL model, recommended is Qwen free model)

    Output:
      response: the LLM response to the query, extracted from the API response

    Preconditions:
      OPENROUTER_API_KEY must be set as an environment variable and contain a valid API key for OpenRouter
  '''

  # Get API key from env variables
  OPENROUTER_API_KEY = os.environ['OPENROUTER_API_KEY']
  if not OPENROUTER_API_KEY:
      raise PermissionError("API key `OPENROUTER_API_KEY` not found in environment variables")
  
  # Confirm that number of text documents corresponds to number of priorities
  if len(documents_text) != len(documents_priorities):
     raise ValueError(f"Unequal number of documents ({len(documents_text)}) and priorities ({len(documents_priorities)})")
  
  # Open a connection to OpenRouter
  client = OpenAI(
    base_url="https://openrouter.ai/api/v1",    # Using the OpenRouter API
    api_key=OPENROUTER_API_KEY
  )
  
  # Start composing the message
  messages = [
      {
        "role": "user",
        "content": []
      }]
  
  # Add images to message content
  for b64_image_url in b64_image_urls:
    messages[0]["content"].append(
      {
        "type": "image_url",
        "image_url": {
          "url": b64_image_url
        }
      }
    )

  # Proper image format for b64_image_urls (TODO: implement this)
  # "type": "image_url",
  # Pass in Base64 image data. 
  # Note that the image format (i.e., image/{format}) needs to be consistent with the Content Type in the supported image list.
  # PNG image:  f"data:image/png;base64,{base64_image}"
  # JPEG image: f"data:image/jpeg;base64,{base64_image}"
  # WEBP image: f"data:image/webp;base64,{base64_image}"
  # "image_url": {"url": f"data:image/png;base64,{base64_image}"},
               

  # Create the query using text from the document
  full_query_with_context = ""
  for i, document_text in enumerate(documents_text):
    full_query_with_context += f"DOCUMENT {i}: \n{document_text}\n\n"

  full_query_with_context += f"QUESTION:\n{user_query}\n\n"

  full_query_with_context += "INSTRUCTIONS: Answer the user's QUESTION using the text from the DOCUMENTS and content of the IMAGES above. Keep your answer ground in the facts of the DOCUMENTS. If the DOCUMENTS do not contain enough information to answer the QUESTION, respond with 'The provided documents do not contain sufficient information to answer this question.'"

  messages[0]["content"].append(
    {
      "type": "text",
      "text": full_query_with_context
    }
  )

  print(f"Querying {model_to_use}")
  print(f"Query: {messages}")

  # Actually query the model
  completion = client.chat.completions.create(
    extra_body={},
    model=model_to_use,
    messages=messages,
    temperature=0
  )
  return completion.choices[0].message.content


# Demo run
if __name__=="__main__":
   q = "Would George like a pickle? How about Lucy?"

   documents = [
      "Hi! My name is George and I like Pickles, my cat. Not the food, though.",
      "Hi! My name is Lucy and I like pickles and my cat."
   ]

   doc_priorities = [
      5,
      1
   ]

    # Test later
   images = []

   response = query(user_query=q, documents_text=documents, documents_priorities=doc_priorities, b64_image_urls=images)
   print(response)
