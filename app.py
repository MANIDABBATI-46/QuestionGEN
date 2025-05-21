# app.py - FULL UPDATED CODE
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
import io
import os # Import os for environment variables and secrets
import secrets # For generating secure initial admin password

from flask_bootstrap import Bootstrap
import spacy
from collections import Counter
import random
from PyPDF2 import PdfReader
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import datetime
from io import BytesIO
from xhtml2pdf import pisa
from bson.objectid import ObjectId

APP_NAME = "QuestionGEN"
TAGLINE = "Transforming Text2MCQ's with NLP."
app = Flask(__name__)
app.name = "QuestionGEN"
Bootstrap(app)

# IMPORTANT: Use an environment variable for secret key in production!
# For development, you can set it like this: os.environ.get('FLASK_SECRET_KEY', 'YOUR_STRONG_RANDOM_KEY_HERE')
# In production, ensure FLASK_SECRET_KEY is set in your environment.
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_very_strong_and_unique_random_secret_key_that_you_should_change_in_production_12345')

# MongoDB Setup
client = MongoClient('mongodb://localhost:27017/')
db = client['NLP_QuestionGEN']
users_collection = db['users']
history_collection = db['history']

# Load spaCy model
nlp = None
use_similarity = False
try:
    nlp = spacy.load("en_core_web_md")
    use_similarity = True
    print("Loaded spaCy model: en_core_web_md")
except OSError:
    print("en_core_web_md not found, attempting to load en_core_web_sm...")
    try:
        nlp = spacy.load("en_core_web_sm")
        print("Loaded spaCy model: en_core_web_sm")
    except OSError:
        print("Neither en_core_web_md nor en_core_web_sm found. Please install a spaCy model (e.g., 'python -m spacy download en_core_web_sm'). MCQ generation will be limited.")
        print("NLP functionalities will be severely restricted or unavailable.")


# --- Utility Functions ---

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session: # Use 'user_id' for consistency
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session: # Use 'user_id' for consistency
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        
        # Fetch user details to ensure role is current and accurate
        user = users_collection.find_one({'_id': ObjectId(session['user_id'])})
        if not user or user.get('role') != 'admin': # Check actual role from DB, not just session
            flash('Access denied. Admin privileges required.', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

def is_valid_option_text(option_text, exclude_list=[]):
    """
    Checks if an option text is valid (not too short, not just numbers/punctuation, not in exclude_list).
    """
    if not option_text or len(option_text.strip()) < 2:
        return False
    # Avoid options that are purely numeric or special characters
    if not any(c.isalpha() for c in option_text):
        return False
    if option_text.lower() in [s.lower() for s in exclude_list]:
        return False
    return True

def generate_mcqs(text, num_questions):
    if nlp is None:
        flash("NLP model not loaded. Cannot generate MCQs.", 'danger')
        return []

    doc = nlp(text)
    # Filter for sentences with a reasonable length to form questions
    sentences = [sent.text for sent in doc.sents if len(sent.text.split()) > 5]
    mcqs = []

    # Iterate through potential sentences, trying to generate 'num_questions'
    # Use a set to prevent duplicate questions if a sentence yields multiple valid answers
    processed_sentences = set()

    attempts = 0
    max_attempts_per_question = 5 # Try a few times to get unique questions/answers
    while len(mcqs) < num_questions and attempts < len(sentences) * max_attempts_per_question:
        if not sentences:
            break # No more sentences to process

        sent = random.choice(sentences)
        sentences.remove(sent) # Remove to avoid reprocessing same sentence immediately
        processed_sentences.add(sent) # Add to general processed set

        doc_sent = nlp(sent)
        # Prioritize Nouns, Proper Nouns, or Entities for answers
        possible_answers = [ent.text for ent in doc_sent.ents if ent.label_ not in ["CARDINAL", "ORDINAL", "DATE", "TIME", "MONEY", "PERCENT", "QUANTITY"]]
        if not possible_answers:
            possible_answers = [token.text for token in doc_sent if token.pos_ in ["NOUN", "PROPN"] and is_valid_option_text(token.text)]

        # Filter out very short or non-descriptive answers
        possible_answers = list(set([ans for ans in possible_answers if is_valid_option_text(ans)]))

        if not possible_answers:
            attempts += 1
            continue

        answer = random.choice(possible_answers)
        
        # Create question stem by replacing the answer with a blank
        question_text = sent.replace(answer, "_______", 1)

        options = []
        options_texts = set()
        options_texts.add(answer) # Always include the correct answer

        # Generate distractors
        distractor_candidates = []
        
        # 1. Similarity-based distractors (if model supports it)
        if use_similarity and nlp(answer).has_vector:
            for token in doc: # Search in the whole document for relevant nouns/proper nouns
                if token.text.lower() == answer.lower() or not is_valid_option_text(token.text):
                    continue
                if token.pos_ in ["NOUN", "PROPN"] and nlp(token.text).has_vector:
                    try:
                        similarity = nlp(answer).similarity(nlp(token.text))
                        if 0.4 < similarity < 0.8: # Tune similarity range as needed
                            distractor_candidates.append(token.text)
                    except Exception:
                        pass # Handle potential issues with similarity calculation

        # 2. Other nouns/proper nouns from the text
        all_nouns_and_proper_nouns = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"] and is_valid_option_text(token.text)]
        distractor_candidates.extend(all_nouns_and_proper_nouns)
        
        # Remove duplicates and the correct answer from candidates
        distractor_candidates = list(set([d for d in distractor_candidates if is_valid_option_text(d, exclude_list=list(options_texts))]))
        random.shuffle(distractor_candidates) # Shuffle for randomness

        # Select distractors, ensuring uniqueness
        for distractor in distractor_candidates:
            if len(options_texts) >= 4:
                break
            if is_valid_option_text(distractor, exclude_list=list(options_texts)):
                options_texts.add(distractor)
        
        # Fallback: If not enough unique distractors, pick random words (less ideal)
        while len(options_texts) < 4:
            random_word_candidates = [token.text for token in doc if len(token.text) > 2 and token.pos_ not in ["PUNCT", "SPACE", "DET", "ADP", "AUX", "VERB"] and is_valid_option_text(token.text, exclude_list=list(options_texts))]
            if not random_word_candidates:
                break # Cannot find more words
            
            random_word = random.choice(random_word_candidates)
            if is_valid_option_text(random_word, exclude_list=list(options_texts)):
                options_texts.add(random_word)

        # If after all attempts we still don't have enough options, skip this question
        if len(options_texts) < 4:
            attempts += 1
            continue

        # Shuffle options and assign labels (A, B, C, D)
        shuffled_options = list(options_texts)
        random.shuffle(shuffled_options)

        final_mcq_options = []
        correct_answer_label = ''
        for idx, opt_text in enumerate(shuffled_options):
            label = chr(65 + idx)
            final_mcq_options.append({'label': label, 'text': opt_text})
            if opt_text == answer:
                correct_answer_label = label
        
        # Add the MCQ if a correct answer label was found (should always be if 'answer' is in options_texts)
        if correct_answer_label:
            mcqs.append({
                'question': question_text,
                'options': final_mcq_options,
                'answer': correct_answer_label, # Stored as 'A', 'B', 'C', 'D'
                'original_answer_text': answer # Stored for reference/debugging
            })
        attempts += 1

    return mcqs

def render_pdf(html_content):
    """Renders an HTML string to a PDF using xhtml2pdf."""
    result_file = BytesIO()
    pisa_status = pisa.CreatePDF(
            html_content,
            dest=result_file)
    if pisa_status.err:
        return None, "Error generating PDF"
    
    result_file.seek(0)
    return result_file, None # Return the BytesIO object and None for error

# --- User Authentication Routes ---
@app.route('/')
def home():
    if 'user_id' in session:
        if session.get('user_role') == 'admin':
            return redirect(url_for('admin_dashboard'))
        else:
            return redirect(url_for('index'))
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST']) # Removed trailing slash for consistency if not needed
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if not username or not password or len(username) < 4:
            flash('Invalid username or password. Username must be at least 4 characters long and password cannot be empty.', 'danger')
            return render_template('signup.html')

        if users_collection.find_one({'username': username}):
            flash('Username already exists. Please choose a different one.', 'danger')
            return render_template('signup.html')

        hashed_password = generate_password_hash(password)
        users_collection.insert_one({'username': username, 'password': hashed_password, 'role': 'user'})
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST']) # Removed trailing slash
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users_collection.find_one({'username': username})

        if user and check_password_hash(user['password'], password):
            session['user_id'] = str(user['_id'])
            session['username'] = user['username']
            session['user_role'] = user.get('role', 'user') # Default to 'user' if role not set
            flash(f'Welcome, {username}!', 'success')
            if session['user_role'] == 'admin':
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'danger')
            return render_template('login.html') # Render login page again with error
    return render_template('login.html')

@app.route('/logout') # Removed trailing slash
@login_required
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

# --- MCQ Generation and Display ---
@app.route('/index', methods=['GET']) # Only GET method allowed for rendering the main page
@login_required
def index():
    # This route is for displaying the form. Actual generation handled by generate_mcqs_route.
    return render_template('index.html')

@app.route('/generate_mcqs', methods=['POST']) # Removed trailing slash
@login_required
def generate_mcqs_route():
    text_input = request.form.get('text_input', '')
    num_questions = int(request.form.get('num_questions', 5))
    uploaded_files = request.files.getlist('files[]')
    
    if not text_input and not uploaded_files:
        flash('Please provide text or upload a file to generate MCQs.', 'danger')
        return redirect(url_for('index'))

    content_text = text_input
    if uploaded_files:
        for file in uploaded_files:
            if file.filename == '':
                continue
            
            file_extension = file.filename.rsplit('.', 1)[1].lower()
            if file_extension == 'pdf':
                try:
                    reader = PdfReader(file)
                    for page in reader.pages:
                        content_text += page.extract_text() + "\n"
                except Exception as e:
                    flash(f"Error processing PDF file: {file.filename} - {e}", 'danger')
                    return redirect(url_for('index'))
            elif file_extension == 'txt':
                try:
                    content_text += file.read().decode('utf-8') + "\n"
                except Exception as e:
                    flash(f"Error processing TXT file: {file.filename} - {e}", 'danger')
                    return redirect(url_for('index'))
            else:
                flash(f"Unsupported file type: {file.filename}. Only PDF and TXT are supported.", 'danger')
                return redirect(url_for('index'))

    if not content_text.strip():
        flash('No usable text extracted from input or files. Please provide more content.', 'danger')
        return redirect(url_for('index'))

    mcqs = generate_mcqs(content_text, num_questions)

    if not mcqs:
        flash('Could not generate MCQs from the provided text. Please try with different content or ensure spaCy model is loaded.', 'warning')
        return redirect(url_for('index'))

    # Store history
    history_item = {
        'user_id': session['user_id'], # Store as string, convert to ObjectId when querying
        'username': session['username'],
        'timestamp': datetime.datetime.now(),
        'input_content': content_text[:500] + '...' if len(content_text) > 500 else content_text, # Store snippet
        'mcqs': mcqs, # Store generated MCQs
        'quiz_score': None, # Initialize quiz score as None
        'quiz_details': [] # Initialize quiz details (for specific answers in quiz)
    }
    inserted_id = history_collection.insert_one(history_item).inserted_id
    session['current_history_item_id'] = str(inserted_id) # Store ID for later quiz result update

    session['current_mcqs'] = mcqs # Store current MCQs in session for quiz and PDF generation
    
    return render_template('mcqs.html', mcqs=mcqs)


@app.route('/download_pdf') # Removed trailing slash
@login_required
def download_pdf():
    mcqs = session.get('current_mcqs')
    if not mcqs:
        flash('No MCQs available to download.', 'warning')
        return redirect(url_for('index'))
    
    # Render PDF template without answers
    rendered_html = render_template('pdf_template.html', mcqs=mcqs, include_answers=False)
    
    pdf_file_buffer, err = render_pdf(rendered_html)
    if pdf_file_buffer:
        return send_file(pdf_file_buffer, download_name="mcq_questions.pdf", as_attachment=True, mimetype='application/pdf')
    else:
        flash(f'Error generating PDF: {err}', 'danger')
        return redirect(url_for('index')) # Redirect to index if error


@app.route('/download_pdf_with_answers') # Removed trailing slash
@login_required
def download_pdf_with_answers():
    mcqs = session.get('current_mcqs')
    if not mcqs:
        flash('No MCQs available to download.', 'warning')
        return redirect(url_for('index'))

    # Render PDF template with answers
    rendered_html = render_template('pdf_template.html', mcqs=mcqs, include_answers=True)
    
    pdf_file_buffer, err = render_pdf(rendered_html)
    if pdf_file_buffer:
        return send_file(pdf_file_buffer, download_name="mcq_questions_with_answers.pdf", as_attachment=True, mimetype='application/pdf')
    else:
        flash(f'Error generating PDF: {err}', 'danger')
        return redirect(url_for('index')) # Redirect to index if error


# --- Quiz Mode ---
@app.route('/start_quiz') # Removed trailing slash
@login_required
def start_quiz():
    mcqs = session.get('current_mcqs')
    if not mcqs:
        flash('No MCQs available to start a quiz. Generate some first!', 'warning')
        return redirect(url_for('index'))
    
    # Randomize the order of questions for the quiz
    quiz_mcqs = list(mcqs) # Create a copy to shuffle
    random.shuffle(quiz_mcqs)

    session['quiz_questions'] = quiz_mcqs
    session['quiz_index'] = 0
    session['quiz_score'] = 0
    session['quiz_answers'] = [] # To store user's specific choices for results display

    question_index = session['quiz_index']
    if question_index < len(session['quiz_questions']):
        question = session['quiz_questions'][question_index]
        return render_template('quiz.html', question=question, question_number=question_index + 1, total_questions=len(quiz_mcqs))
    else:
        # This case should ideally not be reached if called from index
        flash('No questions found for the quiz. Please generate MCQs first.', 'warning')
        return redirect(url_for('index'))

@app.route('/submit_quiz', methods=['POST']) # Removed trailing slash
@login_required
def submit_quiz():
    quiz_questions = session.get('quiz_questions')
    quiz_index = session.get('quiz_index', 0)
    quiz_score = session.get('quiz_score', 0)
    quiz_answers = session.get('quiz_answers', [])

    if not quiz_questions or quiz_index >= len(quiz_questions):
        flash('Quiz already finished or no questions available.', 'danger')
        return redirect(url_for('index'))

    current_question_data = quiz_questions[quiz_index]
    user_answer_label = request.form.get('answer') # This will be 'A', 'B', 'C', 'D'

    if user_answer_label is None:
        flash("Please select an answer before submitting.", 'warning')
        # Re-render the current question
        return render_template('quiz.html',
                               question=current_question_data,
                               question_number=quiz_index + 1,
                               total_questions=len(quiz_questions))

    is_correct = (user_answer_label == current_question_data['answer']) # 'answer' stores 'A', 'B', 'C', 'D'
    if is_correct:
        quiz_score += 1

    quiz_answers.append({
        'question': current_question_data['question'],
        'options': current_question_data['options'], # List of {'label': 'A', 'text': 'Opt Text'}
        'user_choice': user_answer_label,
        'correct_answer_label': current_question_data['answer'], # The label 'A', 'B', 'C', 'D'
        'is_correct': is_correct
    })

    session['quiz_score'] = quiz_score
    session['quiz_answers'] = quiz_answers
    session['quiz_index'] = quiz_index + 1

    if session['quiz_index'] < len(quiz_questions):
        next_question = quiz_questions[session['quiz_index']]
        return render_template('quiz.html', question=next_question, question_number=session['quiz_index'] + 1, total_questions=len(quiz_questions))
    else:
        # Quiz finished, save results to history item that generated the MCQs
        history_item_id = session.get('current_history_item_id') # Get ID from when MCQs were generated
        if history_item_id:
            try:
                history_collection.update_one(
                    {'_id': ObjectId(history_item_id), 'user_id': session['user_id']},
                    {'$set': {
                        'quiz_score': f"{quiz_score}/{len(quiz_questions)}", # Store as "X/Y"
                        'quiz_details': quiz_answers,
                        'quiz_timestamp': datetime.datetime.now() # Store when the quiz was completed
                    }},
                    upsert=False
                )
            except Exception as e:
                print(f"Error updating quiz results in history: {e}")
                flash("Error saving quiz results to history.", 'danger')

        # Clear quiz-related session variables
        session.pop('quiz_questions', None)
        session.pop('quiz_index', None)
        session.pop('quiz_score', None)
        session.pop('quiz_answers', None)
        session.pop('current_history_item_id', None) # Clear the history ID that linked to this quiz

        return redirect(url_for('quiz_results'))

@app.route('/quiz_results') # Removed trailing slash
@login_required
def quiz_results():
    user_id = session['user_id']
    
    # Find the most recent history item for the user that has quiz results
    # Sort by quiz_timestamp in descending order to get the latest completed quiz
    latest_quiz_result = history_collection.find_one(
        {'user_id': user_id, 'quiz_score': {'$ne': None}}, # 'quiz_score' not None means a quiz was completed
        sort=[('quiz_timestamp', -1)]
    )

    if latest_quiz_result:
        quiz_score_raw = latest_quiz_result.get('quiz_score') # e.g., "5/10"
        quiz_details = latest_quiz_result.get('quiz_details', [])

        correct_count = 0
        total_questions = 0 # Initialize total_questions

        if isinstance(quiz_score_raw, str) and '/' in quiz_score_raw:
            try:
                score_parts = quiz_score_raw.split('/')
                correct_count = int(score_parts[0])
                total_questions = int(score_parts[1])
            except ValueError:
                # If parsing fails, fall back to counts from quiz_details if available
                if len(quiz_details) > 0:
                    correct_count = sum(1 for q in quiz_details if q.get('is_correct'))
                    total_questions = len(quiz_details)
                else:
                    correct_count = 0
                    total_questions = 0
        elif quiz_score_raw is not None:
            # If quiz_score_raw is not a string (e.g., an int), treat it as correct_count
            try:
                correct_count = int(quiz_score_raw)
                if len(quiz_details) > 0:
                    total_questions = len(quiz_details)
                else:
                    total_questions = 0 # Unknown total if no details
            except ValueError:
                correct_count = 0
                total_questions = 0


        # Ensure total_questions always reflects len(quiz_details) if quiz_details is available
        if len(quiz_details) > 0 and total_questions != len(quiz_details):
             total_questions = len(quiz_details)
             # Re-calculate correct_count based on quiz_details if inconsistent
             correct_count = sum(1 for q in quiz_details if q.get('is_correct'))


        return render_template('quiz_results.html',
                               correct_answers=correct_count, # Pass individual counts
                               total_questions=total_questions, # Pass individual counts
                               quiz_details=quiz_details)
    else:
        flash("No quiz results found. Please complete a quiz.", 'warning')
        return redirect(url_for('index'))

# --- History ---
@app.route('/history') # Removed trailing slash
@login_required
def history():
    user_id = session['user_id']
    # Convert cursor to list to enable |length filter in Jinja2
    user_history = list(history_collection.find({'user_id': user_id}).sort('timestamp', -1))
    return render_template('history.html', history=user_history)

@app.route('/delete_history_item/<item_id>', methods=['POST']) # Removed trailing slash
@login_required
def delete_history_item(item_id):
    """Deletes a specific history item for the logged-in user."""
    user_id = session['user_id']
    try:
        result = history_collection.delete_one({'_id': ObjectId(item_id), 'user_id': user_id})
        if result.deleted_count == 1:
            flash('History item deleted successfully.', 'success')
        else:
            flash('History item not found or you do not have permission to delete it.', 'danger')
    except Exception as e:
        flash(f"Error deleting history item: {e}", 'danger')
    return redirect(url_for('history'))

# --- Admin Routes ---
@app.route('/admin/dashboard') # Removed trailing slash
@admin_required
def admin_dashboard():
    total_users = users_collection.count_documents({})
    total_mcq_generations = history_collection.count_documents({})
    
    # Get recent generations for display on admin dashboard
    recent_generations = list(history_collection.find({}).sort('timestamp', -1).limit(5))
    
    return render_template('admin_dashboard.html', 
                            total_users=total_users, 
                            total_mcq_generations=total_mcq_generations,
                            recent_generations=recent_generations)

@app.route('/admin/users') # Removed trailing slash
@admin_required
def admin_users():
    users = list(users_collection.find({}).sort('username', 1))
    return render_template('admin_users.html', users=users)

@app.route('/admin/users/delete/<user_id>', methods=['POST']) # Removed trailing slash
@admin_required
def admin_delete_user(user_id):
    if session['user_id'] == user_id:
        flash('You cannot delete your own admin account.', 'danger')
        return redirect(url_for('admin_users'))
    
    try:
        user_obj_id = ObjectId(user_id)
        # Delete user's history first
        history_collection.delete_many({'user_id': user_id}) # Ensure this is user_id string if stored as such
        # Then delete the user
        result = users_collection.delete_one({'_id': user_obj_id})
        if result.deleted_count == 1:
            flash('User and their history deleted successfully.', 'success')
        else:
            flash('User not found.', 'danger')
    except Exception as e:
        flash(f'Error deleting user: {e}', 'danger')
    return redirect(url_for('admin_users'))

@app.route('/admin/users/toggle_admin/<user_id>', methods=['POST']) # Removed trailing slash
@admin_required
def admin_toggle_admin(user_id):
    if session['user_id'] == user_id:
        flash('You cannot change your own admin status.', 'danger')
        return redirect(url_for('admin_users'))

    try:
        user_obj_id = ObjectId(user_id)
        user = users_collection.find_one({'_id': user_obj_id})
        if user:
            new_role = 'user' if user.get('role') == 'admin' else 'admin'
            users_collection.update_one(
                {'_id': user_obj_id},
                {'$set': {'role': new_role}}
            )
            flash(f'User {user["username"]} role changed to {new_role}.', 'success')
        else:
            flash('User not found.', 'danger')
    except Exception as e:
        flash(f'Error toggling admin status: {e}', 'danger')
    return redirect(url_for('admin_users'))

@app.route('/admin/history') # Removed trailing slash
@admin_required
def admin_history():
    all_history_items = list(history_collection.find().sort('timestamp', -1)) # Convert to list
    return render_template('admin_history.html', history=all_history_items)

@app.route('/admin/history/delete/<item_id>', methods=['POST']) # Removed trailing slash
@admin_required
def admin_delete_global_history_item(item_id):
    try:
        result = history_collection.delete_one({'_id': ObjectId(item_id)})
        if result.deleted_count == 1:
            flash('History item deleted successfully.', 'success')
        else:
            flash('History item not found.', 'danger')
    except Exception as e:
        flash(f'Error deleting history item: {e}', 'danger')

    return redirect(url_for('admin_history'))

@app.route('/admin/history/delete_all', methods=['POST']) # Removed trailing slash
@admin_required
def admin_delete_all_history():
    try:
        result = history_collection.delete_many({})
        flash(f'All history ({result.deleted_count} items) deleted successfully.', 'success')
    except Exception as e:
        flash(f'Error deleting all history: {e}', 'danger')

    return redirect(url_for('admin_history'))

if __name__ == '__main__':
    # --- IMPORTANT PRODUCTION SECURITY NOTE ---
    # The following block is for initial setup ONLY in development.
    # In a production environment, you should remove this block or implement
    # a more secure and robust way to create an initial admin user (e.g., via CLI).
    # Hardcoding passwords and checking for admin existence like this is INSECURE.
    # --- END IMPORTANT PRODUCTION SECURITY NOTE ---
    
    if users_collection.find_one({'role': 'admin'}) is None:
        admin_username = 'admin'
        # Generate a secure random password for initial setup if not provided via env
        admin_password = os.environ.get('INITIAL_ADMIN_PASSWORD', secrets.token_urlsafe(16))
        print(f"Creating initial admin user: {admin_username} with password: {admin_password}")
        print("!!! IMPORTANT: Change this password immediately after first login and remove this block in production !!!")
        
        hashed_admin_password = generate_password_hash(admin_password)
        users_collection.insert_one({'username': admin_username, 'password': hashed_admin_password, 'role': 'admin'})

    app.run(debug=True) # Set debug=False for production!