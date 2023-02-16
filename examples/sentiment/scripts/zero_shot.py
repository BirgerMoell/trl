from transformers import pipeline
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

candidate_labels = ["well-being", "non-judgmental", "empathetic", "tailored", "privacy", "crisis", "ethical"]

therapy_text = """Psychologist: Hello, it's nice to meet you. My name is Dr. Johnson, and I'm a psychologist. How are you feeling today?

Patient: Hi, nice to meet you too. I'm feeling a little bit nervous, to be honest.

Psychologist: That's understandable. It can be nerve-wracking to come to therapy for the first time. Can you tell me a bit about what brings you here today?

Patient: Well, I've been feeling really down lately. I'm having trouble sleeping, and I just don't feel like myself. I think I might be depressed.

Psychologist: I'm sorry to hear that you're struggling. It takes a lot of courage to reach out for help. I'm here to listen to you and support you as we work together to find ways to help you feel better. Can you tell me more about what's been going on in your life lately?

Patient: Sure. I lost my job a few months ago, and since then I've just been feeling really lost. I don't know what I want to do with my life, and I'm having trouble finding the motivation to do anything.

Psychologist: Losing a job can be a really difficult experience. It's understandable that you're feeling lost and unmotivated. Have you been able to talk to anyone else about how you're feeling?

Patient: Not really. I don't want to burden my family and friends with my problems.

Psychologist: I understand that you might not want to burden your loved ones, but it's important to have a support system when you're going through a tough time. That's one of the things we can work on together in therapy - building a support network that can help you through this.

Patient: Okay, that sounds good. What else can we do in therapy?

Psychologist: Well, there are many different approaches to therapy, and we'll work together to find what works best for you. One thing we can do is explore your thoughts and feelings to better understand what might be contributing to your depression. We can also work on developing coping skills and strategies to help you manage your symptoms.

Patient: That sounds helpful. I'm willing to try anything to feel better.

Psychologist: That's great to hear. Remember, therapy is a collaborative process, and I'm here to support you as we work together to help you feel better. We'll take things at your pace and work towards your goals. Is there anything else you want to talk about today, or anything you want to know about the therapy process?

Patient: No, I think that's it for now. Thank you for listening and for helping me.

Psychologist: Of course. It's my pleasure to be here for you. I look forward to working with you and helping you on your journey towards healing and wellness."""


classification = classifier(therapy_text, candidate_labels, multi_label=True)
print(classification)