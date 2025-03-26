import os
import numpy as np
import torch
from flask import Flask, request, redirect, url_for, render_template_string
from sentence_transformers import util
import fitz
import base64
from openai import AuthenticationError

from rag_backend import (
    documents, all_chunks, document_counter,
    process_pdf, process_text_file, embedding_model, device
)
import rag_backend  # for updating the document_counter
import chat

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


## ALL ROUTES:
# '/' (index):                  
#   GET is homepage, 
#   POST is posting a query

# '/upload' (upload):           
#   GET is page to upload docs, 
#   POST runs RAG backend for PDF or text file

# '/documents (documents_list): 
#   GET is page to view and edit uploaded docs

# '/update_priority/<int:doc_id>' (update_priority):
#   GET opens form to update priority
#   POST updates priority for doc and redirects to doc list

# '/source/<int:doc_id>/<int:page_number>' (view_source):
#   GET displays a specific page from a specific PDF document as a base64 image


# Base HTML template using Bootstrap and Markdown-Tag
base_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
  <style>
    body {{ padding-top: 2rem; }}
    .container {{ max-width: 800px; }}
    .result {{ margin-bottom: 1.5rem; }}
    footer {{ margin-top: 2rem; text-align: center; font-size: 0.9em; color: #777; }}
  </style>
</head>
<body>
  <div class="container">
    <h1 class="mb-4">{title}</h1>
    {content}
    <footer>
      <p>Local RAG Web Interface &copy; 2025</p>
    </footer>
  </div>
</body>
</html>
<script src="https://cdn.jsdelivr.net/gh/MarketingPipeline/Markdown-Tag/markdown-tag.js"></script> 
'''


def render_page(title, content):
    """Helper: renders a full page with the base template."""
    return render_template_string(base_template.format(title=title, content=content))


@app.route('/', methods=['GET', 'POST'])
def index():
    # Collect the HTML to render as a string based on current state
    results_html = ""
    if request.method == 'POST':
        query = request.form.get('query')
        if query:
            if not all_chunks:
                results_html += "<div class='alert alert-warning'>No documents uploaded yet.</div>"
            else:
                # Stack embeddings into a tensor
                embeddings = np.stack([chunk["embedding"] for chunk in all_chunks])
                embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
                query_embedding = embedding_model.encode(query, convert_to_tensor=True)
                dot_scores = util.dot_score(query_embedding, embeddings_tensor)[0]

                # Adjust score by document priority bonus
                beta = 0.02     # Multiplicative bonus of 2% towards similarity for each priority 1-10
                bonus = torch.tensor([beta * (chunk["priority"]) for chunk in all_chunks],
                                     dtype=torch.float32).to(device)
                adjusted_scores = dot_scores * (1+bonus)
                topk = torch.topk(adjusted_scores, k=min(5, len(all_chunks)))

                # Actually query the Qwen model
                docs = [(all_chunks[i])["chunk_text"] for i in topk.indices.cpu().numpy()]

                try:
                    response = chat.query(
                        user_query=query,
                        documents_text=docs,
                        documents_priorities=topk.values.cpu().numpy(),
                        b64_image_urls=[]
                    )

                    print("\n\n\nRESPONSE:\n\n\n")
                    print(response)

                    # HTML for query response
                    results_html += f'''
                    <div class="card mb-3">
                      <div class="card-body">
                        <h5 class="card-title">Response:</h5>
                        <p class="card-text"><md>{response}</md></p>
                      </div>
                    </div>
                    '''
                except KeyError:
                    results_html += "<div class='alert alert-warning'>Make sure that OPENROUTER_API_KEY is defined in your environment variables.</div>"
                except AuthenticationError:
                    results_html += "<div class='alert alert-warning'>Invalid API Key found in env variables.</div>"
                except:
                    results_html += "<div class='alert alert-warning'>Unable to complete query.</div>"

                # Add query results to HTML
                results_html += f"<h2>Results for query: <em>{query}</em></h2>"
                for score, idx in zip(topk.values.cpu().numpy(), topk.indices.cpu().numpy()):
                    # Get all info about the chunk
                    chunk = all_chunks[idx]

                    # If the document is a PDF, display which page it is from and link the PDF
                    view_source_link = ""
                    if "page_number" in chunk:
                        view_source_link = f'''<a href="{url_for('view_source', doc_id=chunk['doc_id'], page_number=chunk['page_number'])}" class="btn btn-link">View Source</a>'''
                    
                    # Add chunk document, priority, score, text, and link to results
                    results_html += f'''
                    <div class="card mb-3">
                      <div class="card-body">
                        <h5 class="card-title">Document: {chunk["doc_name"]} (Priority: {chunk["priority"]})</h5>
                        <h6 class="card-subtitle mb-2 text-muted">Score: {score:.4f}</h6>
                        <p class="card-text">{chunk["chunk_text"]}</p>
                        {view_source_link}
                      </div>
                    </div>
                    '''

        # No query included in POST
        else:
            results_html += "<div class='alert alert-warning'>Please enter a query.</div>"

    content = '''
    <form method="post" class="mb-4">
      <div class="form-group">
        <label for="query">Enter your query:</label>
        <input type="text" class="form-control" id="query" name="query" placeholder="Type your query here" required>
      </div>
      <button type="submit" class="btn btn-primary">Search</button>
    </form>
    <div class="mb-4">
      <a href="{}" class="btn btn-secondary">Upload Document</a>
      <a href="{}" class="btn btn-secondary">View Documents</a>
    </div>
    <hr>
    <div>{results}</div>
    '''.format(url_for('upload'), url_for('documents_list'), results=results_html)
    return render_page("Local RAG Web Interface", content)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        priority = request.form.get('priority', type=int) or 1
        if not file:
            return "No file uploaded", 400
        filename = file.filename
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        # Update document counter and process file
        doc_id = rag_backend.document_counter
        rag_backend.document_counter += 1
        if filename.lower().endswith('.pdf'):
            process_pdf(save_path, doc_id, filename, priority)
        else:
            process_text_file(save_path, doc_id, filename, priority)
        documents.append({
            "doc_id": doc_id,
            "filename": filename,
            "priority": priority
        })
        return redirect(url_for('documents_list'))
    content = '''
    <h2>Upload Document</h2>
    <form method="post" enctype="multipart/form-data">
      <div class="form-group">
        <label for="file">Choose File:</label>
        <input type="file" class="form-control-file" id="file" name="file" required>
      </div>
      <div class="form-group">
        <label for="priority">Priority (integer, higher means more preferred):</label>
        <input type="number" class="form-control" id="priority" name="priority" value="1" required>
      </div>
      <button type="submit" class="btn btn-primary">Upload</button>
    </form>
    <br>
    <a href="{}" class="btn btn-link">Back to Home</a>
    '''.format(url_for('index'))
    return render_page("Upload Document", content)


@app.route('/documents')
def documents_list():
    content = "<h2>Uploaded Documents</h2><ul class='list-group'>"
    for doc in documents:
        content += f'''
        <li class="list-group-item d-flex justify-content-between align-items-center">
          ID: {doc["doc_id"]} - {doc["filename"]} (Priority: {doc["priority"]})
          <a href="{url_for('update_priority', doc_id=doc["doc_id"])}" class="btn btn-sm btn-outline-primary">Update Priority</a>
        </li>
        '''
    content += "</ul><br><a href='{}' class='btn btn-secondary'>Upload More Documents</a> | <a href='{}' class='btn btn-secondary'>Home</a>".format(
        url_for('upload'), url_for('index'))
    return render_page("Uploaded Documents", content)


@app.route('/update_priority/<int:doc_id>', methods=['GET', 'POST'])
def update_priority(doc_id):
    doc = next((d for d in documents if d["doc_id"] == doc_id), None)
    if not doc:
        return "Document not found", 404
    
    if request.method == 'POST':
        new_priority = request.form.get('priority', type=int)
        doc["priority"] = new_priority
        # Update priority in all chunks corresponding to this document
        for chunk in all_chunks:
            if chunk.get("doc_id") == doc_id:
                chunk["priority"] = new_priority
        return redirect(url_for('documents_list'))
    
    content = f'''
    <h2>Update Priority for Document {doc_id}</h2>
    <form method="post">
      <div class="form-group">
        <label for="priority">New Priority (integer):</label>
        <input type="number" class="form-control" id="priority" name="priority" value="{doc["priority"]}" required>
      </div>
      <button type="submit" class="btn btn-primary">Update</button>
    </form>
    <br>
    <a href="{url_for('documents_list')}" class="btn btn-link">Back to Documents List</a>
    '''
    return render_page("Update Priority", content)


@app.route('/source/<int:doc_id>/<int:page_number>')
def view_source(doc_id, page_number):
    # Find the document info by doc_id
    doc_info = next((d for d in documents if d["doc_id"] == doc_id), None)
    if not doc_info:
        return "Document not found", 404
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], doc_info["filename"])
    pdf_doc = fitz.open(file_path)
    if page_number < 0 or page_number >= pdf_doc.page_count:
        pdf_doc.close()
        return "Page not found", 404
    
    page = pdf_doc[page_number]
    pix = page.get_pixmap(dpi=150)
    pdf_doc.close()

    # Encode image as base64 to embed in HTML
    image_bytes = pix.tobytes("png")
    encoded = base64.b64encode(image_bytes).decode("utf-8")

    # Display the selected page as base64 image in the HTML
    html = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>Source Page - {doc_info["filename"]} Page {page_number + 1}</title>
      <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    </head>
    <body>
      <div class="container mt-4">
        <h2>{doc_info["filename"]} - Page {page_number + 1}</h2>
        <img src="data:image/png;base64,{encoded}" class="img-fluid mb-4" alt="Source page">
        <a href="{url_for('index')}" class="btn btn-primary">Back to Home</a>
      </div>
    </body>
    </html>
    '''
    return html

if __name__ == '__main__':
    app.run(debug=True)
