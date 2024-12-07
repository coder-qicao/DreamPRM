def generate_response(processor, model, prompt, images, max_new_tokens=1000, temperature=0.0, do_sample=False):
    inputs = processor(prompt, images, return_tensors="pt").to("cuda")

    generation_args = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
    }

    generate_ids = model.generate(**inputs,
                                       eos_token_id=processor.tokenizer.eos_token_id,
                                       **generation_args
                                       )
    # remove input tokens
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids,
                                           skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0]
    return response