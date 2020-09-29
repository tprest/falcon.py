PY = python3
AUX = *.pyc *.cprof */*.pyc

test:
	$(PY) test.py

profile:
	rm -f $(AUX)
	rm -rf __pycache__
	touch profile_action.cprof
	$(PY) -m cProfile -o profile_action.cprof profile_action.py
	pyprof2calltree -k -i profile_action.cprof &

clean:
	rm -f $(AUX)
	rm -rf __pycache__ */__pycache__
	@echo "Clean done"
