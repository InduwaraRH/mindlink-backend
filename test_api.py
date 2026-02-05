import requests

# Paste your API key here
GENAI_API_KEY = "AIzaSyBqZqvjONLqrcjslS6v3WPtbKRxEVp_JwY" 

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GENAI_API_KEY}"

print("Checking available models...")
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    print("\n✅ SUCCESS! Here are the models you can use:")
    found_any = False
    for model in data.get('models', []):
        # We only care about models that can generate text (chat)
        if "generateContent" in model.get("supportedGenerationMethods", []):
            print(f"  • {model['name']}") # This is the string we need!
            found_any = True
    
    if not found_any:
        print("\n⚠️ No chat models found. Your key might be restricted.")
else:
    print(f"\n❌ Error: {response.status_code}")
    print(response.text)