def split_step(s_id, response):
    s = f"Step {s_id}"
    s_next = f"Step {s_id+1}"
    if s_next in response:
        assistant = response.split(s_next)[0]
    elif "Answer" in response and s in response:
        assistant = response.split("Answer")[0]
    else:
        assistant = ""
    return assistant