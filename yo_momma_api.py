from textgenrnn import textgenrnn

textgen = textgenrnn()
textgen.train_from_file('jokes.txt', new_model=True, num_epochs=500, train_size=0.9, dropout=0.1)
#textgen.generate()

#textgen_2 = textgenrnn('/weights/hacker_news.hdf5')
#textgen_2.generate(3, temperature=1.0)
