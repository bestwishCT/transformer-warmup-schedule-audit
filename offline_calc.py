import json, math
def rate(step,model=512,warmup=4000):
    if step<=0:
        raise ValueError('step must be positive')
    return model**(-0.5)*min(step**(-0.5),step*warmup**(-1.5))
peak=rate(4000)
result={'teaching_only':True,'unit':'learning rate per optimizer step','rows':[{'step':step,'lr':rate(step),'relative_peak':rate(step)/peak,'lr_model1024':rate(step,1024)} for step in [1000,4000,16000]],'width_ratio':rate(4000,1024)/peak,'base_4000_steps_seconds_approx':4000*0.4,'big_4000_steps_seconds_approx':4000}
assert math.isclose(result['rows'][0]['relative_peak'],0.25)
assert math.isclose(result['rows'][2]['relative_peak'],0.5)
print(json.dumps(result,indent=2))
