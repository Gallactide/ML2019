import sys, time
# Decorators
def timer(function):
    # Prints the time a function took
    def wp(*args, **kwargs):
        t = time.time()
        f = function(*args, **kwargs)
        t -= time.time() # Make sure we don't include writing to stdout as part of timing. Ugly, but reliable
        print(" |- Took", -1*t,"seconds to run.")
        return f
    return wp
def print_state(text_before=None, text_after=None, text_before_f=lambda function, *args, **kwargs: "[*] Running "+function.__name__+"...", text_after_f=lambda a,*b,**c:"done."):
    # Handles printing of a functions state
    def wp0(function):
        def wp1(*args, **kwargs):
            _text_before = text_before # Preserve the reference
            if text_before_f and not text_before:
                _text_before = text_before_f(function, *args, **kwargs)
            if not (_text_before[0]=="[" and _text_before[2]=="]"):
                _text_before = "[*] "+_text_before
            sys.stdout.write(_text_before)
            sys.stdout.flush()
            out = function(*args, **kwargs) # Run the actual function
            _text_after = text_after # Preserve the reference
            if text_after_f and not text_after:
                _text_after = text_after_f(out, *args, **kwargs) # Determine after string from now available result
            print("",_text_after) # Flush buffer and prepend default print arg delimiter
            return out
        return wp1
    return wp0
