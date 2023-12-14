import json
import math
import re
import datetime

from django.core.files.base import ContentFile
from django.core.files.images import ImageFile

pi = math.pi
PI = math.pi

from django.db import models
from user.models import User
from .apps import MessageConfig
from chatbot.interface import AnswerHandler, FileManager
import chatbot.logic_engine as le
from django.core.files import File

def transform_string_to_value(input_string):
    try:
        result = eval(input_string)
    except Exception:
        # Define regular expression patterns to match the input string
        patterns = [re.compile(r'([a-zA-Zα-ωΑ-Ω]+)\/(\d*\.?\d*)'),
                    re.compile(r'(\d*\.?\d*)([a-zA-Zα-ωΑ-Ω]+)\/(\d*\.?\d*)'),
                    re.compile(r'(\d*\.?\d*)([a-zA-Zα-ωΑ-Ω]+)'),
                    re.compile(r'([a-zA-Zα-ωΑ-Ω]+)') ]

        # Match the pattern in the input string
        for i, pattern in enumerate(patterns):
            match = pattern.match(input_string)
            if match:
                break

        if i ==0:
            unit = match.group(1)
            denominator = float(match.group(2))

            # Check if the unit is a recognized mathematical constant
            if unit.lower() == 'pi' or unit.lower() == 'π':
                result = math.pi / denominator
            else:
                print(f"Unsupported unit: {unit}")
        elif i == 1:
            numeric_value = float(match.group(1))
            unit = match.group(2)
            denominator = float(match.group(3))

            # Check if the unit is a recognized mathematical constant
            if unit.lower() == 'pi' or unit.lower() == 'π':
                result = numeric_value * math.pi / denominator
            else:
                print(f"Unsupported unit: {unit}")

        elif i == 2:
            if match:
                if input_string.lower() == 'π' :
                    unit = match.group(2)

                    if unit.lower() == 'π':
                        result = math.pi
                    else:
                        print(f"Unsupported unit: {unit}")
                else:
                    numeric_value = float(match.group(1))
                    unit = match.group(2)

                    if unit.lower() == 'pi' or unit.lower() == 'π':
                        result = numeric_value * math.pi
                    else:
                        print(f"Unsupported unit: {unit}")
            else:
                print("Invalid input format")

    return result

def create_more_details_message(previous_message, user):
    more_details_message = Message(content="Sorry, I didn't quite catch that. " +
                                           "Could you provide more details or ask in a different way?",
                                   previous_message=previous_message,
                                   user=user)
    more_details_message.save()

    what_I_do_message = Message(content="Remember, you can ask me about:\n" +
                                        "1. Defining a quantum gate.\n" +
                                        "2. Drawing a quantum gate.\n" +
                                        "3. Applying a quantum gate.\n" +
                                        "Gates include: Identity, Pauli, S, "
                                        "Hadamard, Phase, Rotations, CNOT, CZ, SWAP.\n\n" +
                                        "Let's try again! 🚀✨",
                                   previous_message=more_details_message,
                                   user=user)

    what_I_do_message.save()

    previous_message = what_I_do_message

    return previous_message

class Message(models.Model):
    content = models.TextField(blank=False, null=False)
    role = models.TextField(blank=False, null=False, default="ai")
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='messages')
    previous_message = models.OneToOneField('self', on_delete=models.SET_NULL, null=True, blank=True)
    parameters = models.TextField(blank=True, null=True)
    draw = models.ImageField(verbose_name='draw', max_length=255, blank=True, null=True)


    def __str__(self):
        return f'{self.content} with id = {self.id}'

    def save(self, *args, **kwargs):
        if self.role == 'user':
            super().save(*args, **kwargs)  # Save the original message
            previous_message = self

            if self.content.lower() == 'yes':
                # Chatbot answers that it will proceed with the question
                ai_ok = Message(content="Ok, let's do it!",
                                previous_message=previous_message,
                                user=self.user,
                                parameters=self.parameters)
                ai_ok.save()
                previous_message = ai_ok

                # Extract parameters from the user question
                parameters = json.loads(self.previous_message.parameters)
                category = parameters.get('category')
                user_question = self.previous_message.previous_message.content

                # Save user question and category for retraining
                file_manager = MessageConfig.classifbert.file_manager
                if isinstance(category, int) and user_question:
                    with open(file_manager.folder_path + file_manager.file_name, 'a') as file:
                        file.write(f'{category}\t{user_question}' + '\n')

                gate_name = parameters.get('gate_name')
                gate = le.gates.get(gate_name)

                initial_state_name = parameters.get('initial_state_name')
                initial_state = le.initial_states.get(initial_state_name)

                if gate_name == 'phase':
                    bert_qa = MessageConfig.bert_qa
                    bert_answer = bert_qa.ask_questions(user_question, ["What is the phase shift?"])

                    if bert_answer:
                        try:
                            phase_shift = transform_string_to_value(
                                bert_answer.get('What is the phase shift?').get('answer').get('answer')[0])
                            parameters['phase_shift'] = phase_shift
                        except Exception:
                            phase_shift = 0
                            parameters['phase_shift'] = phase_shift
                            parameters_json = json.dumps(parameters, sort_keys=True, indent=4)
                            error_message = Message(content='The reading for phase shift was unavailable, leading to '
                                                            'the assumption of a phase shift of zero radians.',
                                                    previous_message=previous_message,
                                                    user=self.user,
                                                    parameters=parameters_json)
                            error_message.save()
                            previous_message = error_message
                            print('The reading for phase shift was unavailable, leading to the assumption of a phase '
                                  'shift of zero radians.')
                    else:
                        phase_shift = 0

                    gate = le.PhaseGate(phase_shift)
                elif gate_name == 'rotation':
                    bert_qa = MessageConfig.bert_qa
                    questions = ["What is the angle of the rotation?", "What is the axis of the rotation?"]
                    bert_answer = bert_qa.ask_questions(context=user_question, questions=questions)

                    if bert_answer:
                        try:
                            angle = transform_string_to_value(bert_answer.get(questions[0]).get('answer').get('answer')[0])
                            parameters['angle'] = angle

                        except Exception:
                            print('Rotation angle could not be read, and so an angle of zero radians was assumed.')
                            angle = 0
                            parameters['angle'] = angle
                            parameters_json = json.dumps(parameters, sort_keys=True, indent=4)
                            error_message = Message(
                                content='The rotation angle could not be determined, resulting in the assumption of '
                                        'an angle of zero radians.',
                                previous_message=previous_message,
                                user=self.user,
                                parameters=parameters_json)
                            error_message.save()
                            previous_message = error_message

                        axis = bert_answer.get(questions[1]).get('answer').get('answer')[0].lower()
                        prob = bert_answer.get(questions[1]).get('probability').get('probability')[0]

                        if axis == 'empty' or prob < 0.9:
                            print('The rotation axis reading was unavailable, prompting the assumption of the x-axis '
                                  'as the rotation axis.')
                            axis = 'x'
                            error_message = Message(
                                content='The rotation axis reading was unavailable, prompting the assumption of the '
                                        'x-axis as the rotation axis.',
                                previous_message=previous_message,
                                user=self.user,
                                parameters=parameters)
                            error_message.save()
                            previous_message = error_message

                        parameters['axis'] = axis

                    else:
                        print('Bert did not give any answer...')
                        angle = 0
                        axis = 'x'
                        parameters['angle'] = angle
                        parameters['axis'] = axis
                        error_message = Message(
                            content='The rotation axis and angle readings were unavailable. I assumed the x-axis for '
                                    'the rotation axis and zero radians for the angle.',
                            previous_message=previous_message,
                            user=self.user,
                            parameters=parameters)
                        error_message.save()

                    axis_rotation_map = {'x': 'RX', 'y': 'RY', 'z': 'RZ'}
                    gate_name = axis_rotation_map.get(axis)
                    gate_object = gate.get(gate_name)

                    if gate_object:
                        gate = gate_object(angle)
                    else:
                        gate = le.gates.get('rotation').get('RX')(0)

                answer_handler = AnswerHandler(category, gate, [initial_state])
                le_answer = answer_handler.apply_gate_method()

                if category == 1:
                    parameters_json = json.dumps(parameters, sort_keys=True, indent=4)
                    le_answer_message = Message(content='Here is the circuit:',
                                                previous_message=previous_message,
                                                user=self.user,
                                                parameters=parameters_json)

                    current_date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

                    le_answer_message.draw.save(f'qiskit_draws/{current_date}.png', ContentFile(le_answer), save=True)

                else:
                    parameters_json = json.dumps(parameters, sort_keys=True, indent=4)
                    le_answer_message = Message(content=le_answer,
                                                previous_message=previous_message,
                                                user=self.user,
                                                parameters=parameters_json)

                le_answer_message.save()
                previous_message = le_answer_message

                another_question_message = Message(content="What is your next question?",
                                                   user=self.user,
                                                   previous_message=previous_message)
                Message(content="What is your next question?",
                        user=self.user,
                        previous_message=previous_message).save()
                previous_message = another_question_message

            elif self.content.lower() == 'no':
                previous_message = create_more_details_message(previous_message, self.user)

            else:
                category, logits, top_indices = MessageConfig.classifbert.classify_user_input(self.content)

                gate_name, initial_state_name, understood_question = MessageConfig.classifbert.process_user_question(
                    self.content, category)

                if logits[0][category] < 0.9 or gate_name == None:
                    previous_message = create_more_details_message(previous_message, self.user)

                else:
                    parameters = {"initial_state_name": initial_state_name, 'gate_name': gate_name,
                                      'category': category}
                    parameters_json = json.dumps(parameters, sort_keys=True, indent=4)
                    ai_question_check = Message(content=understood_question,
                                                previous_message=self,
                                                user=self.user,
                                                parameters=parameters_json
                                                )
                    ai_question_check.save()
                    previous_message = ai_question_check
        else:
            super().save(*args, **kwargs)  # Save the new message


