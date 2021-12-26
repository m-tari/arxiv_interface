from . import config_set


def title_generator(abstract, min_length, max_length, model, tokenizer):
	
	# T5 uses a max_length of 512 so we cut the article to 512 tokens.
	inputs = tokenizer("summarize: " + abstract, return_tensors="tf", max_length=512)
	outputs = model.generate(
	    inputs["input_ids"], truncation=True, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True
	)

	title = tokenizer.decode(outputs[0])[5:]

	return title