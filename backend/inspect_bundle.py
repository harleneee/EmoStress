import joblib, json

bundle = joblib.load('models/emostress_best_model_bundle.joblib')
print('=== BUNDLE KEYS ===')
print(list(bundle.keys()))

print('\n=== best_model_name ===')
print(bundle.get('best_model_name'))

print('\n=== model type ===')
m = bundle.get('model')
print(type(m))
if hasattr(m, 'steps'):
    print('Pipeline steps:', [s[0] for s in m.steps])

print('\n=== labels ===')
print(bundle.get('labels'))

print('\n=== feature_columns (first 10) ===')
fc = bundle.get('feature_columns')
if fc is not None:
    print(list(fc)[:10])
    print('Total features:', len(fc))
else:
    print('None')

print('\n=== emotion_to_stress ===')
print(bundle.get('emotion_to_stress'))

print('\n=== metrics ===')
print(json.dumps(bundle.get('metrics', {}), indent=2, default=str))

print('\n=== note ===')
print(bundle.get('note'))

print('\n=== settings ===')
print(bundle.get('settings'))
