import flask as f
import simple_rag as s

app = f.Flask(__name__)
EMBEDDING_MODEL = 'nomic-embed-text'
LLM = 'llama3.2:1b'
chatbot = s.Chatbot(['2025 Helios Intern Handbook.pdf', 'Helios Training 2025.pdf'], EMBEDDING_MODEL, LLM)

@app.route('/', methods=['GET', 'POST'])
def home():
    if f.request.method == 'POST':
        question = f.request.form['question']
        answer = chatbot.answer(question)
        return f.render_template('index.html', answer=answer)
    return f.render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
