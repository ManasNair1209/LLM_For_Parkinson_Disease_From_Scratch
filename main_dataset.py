import json
import random

# Define the structure for stages, descriptions, and recommendations/therapeutic advice
stage_data = {
    "Normal": {
        "description": "The patient is in the Normal stage of Parkinson’s Disease. No abnormal findings; the patient is functioning within normal limits.",
        "recommendations": [
            "Maintain a healthy lifestyle with regular exercise and a balanced diet.",
            "Continue with regular check-ups with your doctor.",
            "Stay mentally active with puzzles, reading, and other cognitive activities.",
            "Engage in activities that you enjoy to maintain a good quality of life.",
            "Ensure you get enough rest and manage stress effectively.",
            "Participate in social activities to maintain connections.",
            "Consider mindfulness or meditation for stress management.",
            "Get adequate sleep to support overall health.",
            "Stay hydrated throughout the day.",
            "Learn about Parkinson's disease to be informed for the future."
        ],
        "therapeutic_advice": [
            "Focus on overall well-being and preventive health measures.",
            "Encourage regular physical activity to maintain fitness.",
            "Suggest cognitive exercises to keep the mind sharp.",
            "Advise on stress reduction techniques.",
            "Promote healthy sleep habits.",
            "Recommend strategies for maintaining social engagement."
        ]
    },
    "Mild PD": {
        "description": "The patient is in the Mild stage of Parkinson’s Disease. The patient displays unilateral symptoms but is still independent.",
        "recommendations": [
            "Consult with a physical therapist to develop an exercise program to improve mobility and balance.",
            "Consider speech therapy to address any early signs of speech changes.",
            "Join a support group to connect with others who are living with Parkinson's.",
            "Explore complementary therapies such as yoga or tai chi to help with flexibility and balance.",
            "Keep a symptom diary to track changes and discuss them with your doctor.",
            "Consider assistive devices like a walking stick for added stability if needed.",
            "Engage in regular aerobic exercise.",
            "Practice voice exercises to maintain vocal strength.",
            "Learn about medication options and potential side effects.",
            "Plan for potential future needs and support."
        ],
        "therapeutic_advice": [
            "Implement a tailored exercise program focusing on flexibility, balance, and strength.",
            "Introduce speech and swallowing exercises if needed.",
            "Provide resources for support groups and patient education.",
            "Recommend aerobic exercise and strength training.",
            "Facilitate access to speech therapy.",
            "Educate on medication management.",
            "Suggest evaluation for assistive devices.",
            "Encourage proactive planning for disease progression."
        ]
    },
    "Moderate PD": {
        "description": "The patient is in the Moderate stage of Parkinson’s Disease. The patient displays bilateral symptoms with postural instability.",
        "recommendations": [
            "Work with an occupational therapist to find ways to make daily activities easier and safer.",
            "Discuss medication options with your doctor to manage your symptoms.",
            "Make home modifications to prevent falls, such as installing grab bars and removing tripping hazards.",
            "Explore adaptive equipment for dressing, eating, and other daily tasks.",
            "Consider enrolling in a Parkinson's specific exercise class.",
            "Ensure you have a clear and safe path to move around your home.",
            "Regularly review and adjust medication timing and dosage.",
            "Work on strategies to improve gait and reduce freezing episodes.",
            "Consider a neurologist specializing in movement disorders.",
            "Assess nutritional needs and swallowing difficulties."
        ],
        "therapeutic_advice": [
            "Assess and address difficulties with daily living activities through occupational therapy.",
            "Optimize medication regimen for symptom control.",
            "Recommend home safety evaluations and fall prevention strategies.",
            "Implement gait training and strategies for freezing.",
            "Facilitate consultation with a movement disorder specialist.",
            "Provide nutritional assessment and support for swallowing issues.",
            "Encourage participation in group exercise programs designed for Parkinson's."
        ]
    },
    "Severe PD": {
        "description": "The patient is in the Severe stage of Parkinson’s Disease. The patient is severely disabled and requires significant assistance.",
        "recommendations": [
            "Consider palliative care to help manage your symptoms and improve your quality of life.",
            "Explore assistive devices, such as a wheelchair or walker, to help with mobility.",
            "Ensure you have a strong support system of family, friends, and caregivers.",
            "Discuss options for in-home care or assisted living facilities.",
            "Focus on maintaining comfort and dignity in all aspects of care.",
            "Ensure proper positioning and skin care to prevent complications.",
            "Explore communication aids if speech becomes significantly impaired.",
            "Develop a routine to manage symptoms and daily activities.",
            "Address non-motor symptoms such as sleep disturbances or mood changes.",
            "Plan for advanced care directives."
        ],
        "therapeutic_advice": [
            "Implement a comprehensive care plan including palliative care services.",
            "Provide appropriate assistive devices and training for mobility.",
            "Support caregivers and provide resources for respite care.",
            "Discuss options for higher level of care.",
            "Prioritize comfort care and symptom management.",
            "Implement strategies to prevent pressure ulcers and other complications of immobility.",
            "Offer psychological and spiritual counseling as needed.",
            "Introduce communication aids if required.",
            "Manage non-motor symptoms with appropriate interventions."
        ]
    },
     "Very Severe PD": {
        "description": "The patient is in the Very Severe stage of Parkinson’s Disease. The patient is completely dependent and bedridden.",
        "recommendations": [
            "Focus on comfort and dignity in all aspects of care.",
            "Ensure proper positioning and skin care to prevent complications.",
            "Provide emotional and spiritual support for the patient and family.",
            "Implement strategies to manage pain and discomfort.",
            "Maintain a calm and peaceful environment.",
            "Ensure regular turning and repositioning.",
            "Provide gentle range of motion exercises.",
            "Offer opportunities for sensory stimulation."
        ],
        "therapeutic_advice": [
            "Prioritize comfort care and symptom management.",
            "Implement strategies to prevent pressure ulcers and other complications of immobility.",
            "Offer psychological and spiritual counseling as needed.",
            "Provide pain management interventions.",
            "Ensure consistent and compassionate care.",
            "Educate caregivers on proper positioning and skin care.",
            "Facilitate access to hospice services if appropriate."
        ]
    }
}

# Generate the new dataset
new_dataset = []
# Create multiple entries for each stage
samples_per_stage = 2000 # Maintain the number of samples per stage for balanced dataset
for stage, data in stage_data.items():
    description = data["description"]
    recommendations = data["recommendations"]
    therapeutic_advice = data["therapeutic_advice"]

    # Combine recommendations and therapeutic advice
    all_advice = recommendations + therapeutic_advice

    # For each stage, create multiple samples with different combined advice
    for _ in range(samples_per_stage):
        advice = random.choice(all_advice)
        # Format the final answer string with the description and a piece of advice
        answer_text = (
            f"{description}\n"
            f"Advice: {advice}" # Using a general "Advice" label
        )

        new_dataset.append({
            "instruction": f"Parkinson's Stage: {stage}",
            "output": {
                "answer": answer_text
            }
        })

# Shuffle the dataset to ensure random distribution for training
random.shuffle(new_dataset)


# Save the new dataset to a JSON file
file_path = "File PATH" # New file name
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(new_dataset, f, indent=4)

print(f"New dataset with {len(new_dataset)} entries created successfully!")
print(f"File saved as: {file_path}")
print("\nSample entry from the new dataset:")
# Print a sample to verify the format
if new_dataset:
    print(json.dumps(new_dataset[0], indent=4))