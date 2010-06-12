
def mergeFunctionMetadata(f, g):
    """
    Overwrite C{g}'s name and docstring with values from C{f}.  Update
    C{g}'s instance dictionary with C{f}'s.

    To use this function safely you must use the return value. In Python 2.3,
    L{mergeFunctionMetadata} will create a new function. In later versions of
    Python, C{g} will be mutated and returned.

    @return: A function that has C{g}'s behavior and metadata merged from
        C{f}.
    """
    try:
        g.__name__ = f.__name__
    except TypeError:
        try:
            merged = new.function(
                g.func_code, g.func_globals,
                f.__name__, inspect.getargspec(g)[-1],
                g.func_closure)
        except TypeError:
            pass
    else:
        merged = g
    try:
        merged.__doc__ = f.__doc__
    except (TypeError, AttributeError):
        pass
    try:
        merged.__dict__.update(g.__dict__)
        merged.__dict__.update(f.__dict__)
    except (TypeError, AttributeError):
        pass
    merged.__module__ = f.__module__
    return merged

class failure:
    pass

class defer:
    _runningCallbacks = False
    called = 0
    paused = 0
    def __init__(self):
        self.callbacks = []
        
    def addCallbacks(self, callback, callbackArgs=None, callbackKeywords=None):
        assert callable(callback)
        cb = (callback, callbackArgs, callbackKeywords)
        self.callbacks.append(cb)

        if self.called:
            self._runCallbacks()
        return self

    def addCallback(self, callback, *args, **kw):
        return self.addCallbacks(callback, callbackArgs=args,
                                 callbackKeywords=kw)
    
    def callback(self, result):
        self._startRunCallbacks(result)

    def pause(self):
        self.paused = self.paused + 1

    def unpause(self):
        self.paused = self.paused - 1
        if self.paused:
            return
        if self.called:
            self._runCallbacks()
            
    def _continue(self, result):
        self.result = result
        self.unpause()

    def _startRunCallbacks(self, result):
        if self.called:
            raise Exception('AlreadyCalledError')
        self.called = True
        self.result = result
        self._runCallbacks()
        
    def _runCallbacks(self):
        if self._runningCallbacks:
            return
        
        if not self.paused:
            while self.callbacks:
                item = self.callbacks.pop(0)
                callback, args, kw = item[isinstance(self.result, None)]
                args = args or ()
                kw = kw or {}
                try:
                    self._runningCallbacks = True
                    try:
                        self.result = callback(self.result, *args, **kw)
                    finally:
                        self._runningCallbacks = False
                    if isinstance(self.result, defer):
                        self.pause()
                        self.result.addCallback(self._continue)
                        break
                    
                except:
                    self.result = failure()

    def __str__(self):
        cname = self.__class__.__name__
        if hasattr(self, 'result'):
            return "<%s at %s  current result: %r>" % (cname, self, self.result)
        return "<%s at %s>" % (cname, self)
    __repr__ = __str__
try:
    BaseException
except NameError:
    BaseException=Exception

class _DefGen_Return(BaseException):
    def __init__(self, value):
        self.value = value

def returnValue(val):
    raise _DefGen_Return(val)

def _inlineCallbacks(result, g, deferred):
    """
    See L{inlineCallbacks}.
    """
    # This function is complicated by the need to prevent unbounded recursion
    # arising from repeatedly yielding immediately ready deferreds.  This while
    # loop and the waiting variable solve that by manually unfolding the
    # recursion.

    waiting = [True, # waiting for result?
               None] # result

    while 1:
        try:
            # Send the last result back as the result of the yield expression.
            #if isinstance(result, failure):
            #    result = result.throwExceptionIntoGenerator(g)
            #else:
            result = g.send(result)
            
        except StopIteration:
            # fell off the end, or "return" statement
            defer.callback(None)
            return deferred
        except _DefGen_Return, e:
            # returnValue call
            defer.callback(e.value)
            return deferred
        except:
            defer.errback()
            return deferred

        if isinstance(result, defer):
            # a deferred was yielded, get the result.
            def gotResult(r):
                if waiting[0]:
                    waiting[0] = False
                    waiting[1] = r
                else:
                    _inlineCallbacks(r, g, deferred)

            result.addBoth(gotResult)
            if waiting[0]:
                # Haven't called back yet, set flag so that we get reinvoked
                # and return from the loop
                waiting[0] = False
                return deferred

            result = waiting[1]
            # Reset waiting to initial values for next loop.  gotResult uses
            # waiting, but this isn't a problem because gotResult is only
            # executed once, and if it hasn't been executed yet, the return
            # branch above would have been taken.


            waiting[0] = True
            waiting[1] = None


    return deferred

def inlineCallbacks(f):
    """
    WARNING: this function will not work in Python 2.4 and earlier!

    inlineCallbacks helps you write Deferred-using code that looks like a
    regular sequential function. This function uses features of Python 2.5
    generators.  If you need to be compatible with Python 2.4 or before, use
    the L{deferredGenerator} function instead, which accomplishes the same
    thing, but with somewhat more boilerplate.  For example::

        def thingummy():
            thing = yield makeSomeRequestResultingInDeferred()
            print thing #the result! hoorj!
        thingummy = inlineCallbacks(thingummy)

    When you call anything that results in a Deferred, you can simply yield it;
    your generator will automatically be resumed when the Deferred's result is
    available. The generator will be sent the result of the Deferred with the
    'send' method on generators, or if the result was a failure, 'throw'.

    Your inlineCallbacks-enabled generator will return a Deferred object, which
    will result in the return value of the generator (or will fail with a
    failure object if your generator raises an unhandled exception). Note that
    you can't use 'return result' to return a value; use 'returnValue(result)'
    instead. Falling off the end of the generator, or simply using 'return'
    will cause the Deferred to have a result of None.

    The Deferred returned from your deferred generator may errback if your
    generator raised an exception::

        def thingummy():
            thing = yield makeSomeRequestResultingInDeferred()
            if thing == 'I love Twisted':
                # will become the result of the Deferred
                returnValue('TWISTED IS GREAT!')
            else:
                # will trigger an errback
                raise Exception('DESTROY ALL LIFE')
        thingummy = inlineCallbacks(thingummy)
    """
    print f
    def unwindGenerator(*args, **kwargs):
        return _inlineCallbacks(None, f(*args, **kwargs), defer())
    return mergeFunctionMetadata(f, unwindGenerator)

def start_job(generator):
    """Start a job (a coroutine that yield tasks)."""
    def _task_return(result):
        def _advance_generator():
            try:
                new_task = generator.send(result)
            except StopIteration:
                return
            new_task(_task_return)
        # isolate the advance of the coroutine
        gobject.idle_add(_advance_generator)            
    _task_return(None)
    return generator
