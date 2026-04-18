SUPPORT_DISEASES=['Aortic regurgitation', 'Aortic stenosis', 'Bicuspid aortic valve', 'Aortic root dilation','Tricuspid Regurgitation','Mitral regurgitation','Mitral stenosis','Pulmonary regurgitation','Pulmonary artery dilation', 'Pulmonary hypertension','Left-atrial Dilation','Right-atrial Dilation','Atrial septal defect','Left-ventricular Dilation','Left-ventricular apical aneurysm','Left-ventricular diastolic dysfunction','Left-ventricular systolic dysfunction','Right-ventricular Dilation', 'Right-ventricular systolic dysfunction','Ventricular septal defect','Inferior vena cava dilation','Hypertrophic cardiomyopathy','Segmental wall-motion abnormality','Pacemaker in situ','Status post aortic-valve replacement','Status post mitral-valve replacement','Mechanical prosthetic valve (post valve replacement)','Pericardial effusion']
    


SYSTEM_PROMPT="You are an expert echocardiologist. Your task is to diagnose heart conditions from the given echocardiography. Answer yes or no."


QUERY = "Based on the echocardiography, does the patient have <disease>?"