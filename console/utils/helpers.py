
ALLOWED_EXTENSIONS = {'seq', 'jpg'}

def allowed_file(filename):
    """
    Checks that the file extension is within the application allowed extensions.

    Parameters
    ----------
    filename : str
        Uploaded file name

    Returns
    -------
    bool
        Allowed or not allowed extension
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def update_session_subdict(sess, first_key, params):
    for second_key in params.keys():
        sess[first_key][second_key] = params[second_key]
    sess.modified = True