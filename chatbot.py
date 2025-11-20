import json
import os
from typing import Dict, List, Optional, Tuple
import re

# Sentence Transformers for semantic search (embeddings)
SEMANTIC_SEARCH_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SEMANTIC_SEARCH_AVAILABLE = True
except (ImportError, AttributeError, RuntimeError) as e:
    SEMANTIC_SEARCH_AVAILABLE = False
    print(f"⚠ Warning: Sentence transformers not available ({type(e).__name__}). Using basic matching.")
    print("   For better semantic search: pip install sentence-transformers scikit-learn")


class CompanyChatbot:
    def __init__(self, data_file: str = "company_data.json"):
        """Initialize the chatbot with company data"""
        self.data_file = data_file
        self.company_data = self.load_company_data()
        self.embedding_model = None
        self.dataset_chunks = []
        self.chunk_embeddings = None
        self.chunk_metadata = []  # Store metadata for each chunk (type, original_key)
        
        # Extract all data into searchable chunks
        self._build_dataset_chunks()
        
        # Initialize semantic search (embeddings)
        if SEMANTIC_SEARCH_AVAILABLE:
            try:
                print("🔄 Loading semantic search model (first time will download ~420MB)...")
                print("   This may take a few minutes on first run...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("🔄 Computing embeddings for dataset chunks...")
                self.chunk_embeddings = self.embedding_model.encode(
                    self.dataset_chunks,
                    show_progress_bar=True,
                    convert_to_numpy=True
                )
                print(f"✓ Semantic search ready! ({len(self.dataset_chunks)} chunks embedded)")
            except Exception as e:
                print(f"⚠ Warning: Could not load semantic search model: {e}")
                self.embedding_model = None
    
    def load_company_data(self) -> Dict:
        """Load company data from JSON file"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: {self.data_file} not found!")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {self.data_file}")
            return {}
    
    def _build_dataset_chunks(self):
        """Build searchable chunks from company data with proper semantic context"""
        chunks = []
        metadata = []
        data = self.company_data
        
        # Company name - VERY specific chunks that ONLY match explicit name questions
        company_name = data.get('company_name', '')
        if company_name:
            # Create chunks that semantically match ONLY explicit "name" questions
            # Reduced and made extremely specific to avoid matching other question types
            chunks.append(f"What is the company name answer {company_name}")
            chunks.append(f"Company name kya hai answer {company_name}")
            chunks.append(f"Company ka naam kya hai answer {company_name}")
            chunks.append(f"Naam kya hai company ka answer {company_name}")
            chunks.append(f"Company name is {company_name}")
            chunks.append(f"The company name is {company_name}")
            # Reduced from 9 to 6 - only most specific name-related chunks
            metadata.extend([{'type': 'company_name'}] * 6)
        
        # Company description - for "who are you" type questions with more context (Urdu/Hindi/English)
        if data.get('company_description'):
            desc = data.get('company_description')
            chunks.append(f"Company description {desc}")
            chunks.append(f"About the company {desc}")
            chunks.append(f"Who we are {desc}")
            chunks.append(f"About us {desc}")
            chunks.append(f"Who are you {desc}")
            chunks.append(f"Tum kon ho {desc}")
            chunks.append(f"Aap kon hain {desc}")
            chunks.append(f"Tell me about your company description {desc}")
            chunks.append(f"Aapki company kya karti hai description {desc}")
            chunks.append(f"What does your company do description {desc}")
            chunks.append(f"Company overview description {desc}")
            # Add more specific chunks to avoid matching founded year
            chunks.append(f"About company information description {desc}")
            chunks.append(f"Company details description overview {desc}")
            chunks.append(f"Company information description {desc}")
            chunks.append(f"Tell me company description {desc}")
            metadata.extend([{'type': 'company_description'}] * 16)
        
        # Company founded year
        if data.get('founded'):
            founded = data.get('founded')
            chunks.append(f"Company founded in {founded}")
            chunks.append(f"Founded year {founded}")
            chunks.append(f"When was the company founded {founded}")
            chunks.append(f"Aapki company kab bani {founded}")
            chunks.append(f"Company kab bani {founded}")
            metadata.extend([{'type': 'founded'}] * 5)
        
        # Team size
        if data.get('team_size'):
            team_size = data.get('team_size')
            chunks.append(f"Team size {team_size}")
            chunks.append(f"Number of team members {team_size}")
            chunks.append(f"How big is your team {team_size}")
            chunks.append(f"Aapke team mein kitne log hain {team_size}")
            chunks.append(f"Team mein kitne log hain {team_size}")
            metadata.extend([{'type': 'team_size'}] * 5)
        
        # International projects
        if data.get('countries_served'):
            countries = ', '.join(data.get('countries_served', []))
            chunks.append(f"Yes we work with clients from {countries}")
            chunks.append(f"International projects {countries}")
            chunks.append(f"Do you work internationally yes {countries}")
            chunks.append(f"Aap international projects karte hain haan {countries}")
            chunks.append(f"International projects haan {countries}")
            metadata.extend([{'type': 'international'}] * 5)
        
        # Portfolio
        portfolio = data.get('portfolio', {})
        if portfolio.get('portfolio_link'):
            portfolio_link = portfolio.get('portfolio_link')
            portfolio_msg = portfolio.get('portfolio_message_en', '')
            chunks.append(f"Portfolio {portfolio_msg} {portfolio_link}")
            chunks.append(f"Do you have a portfolio yes {portfolio_link}")
            chunks.append(f"Portfolio website {portfolio_link}")
            chunks.append(f"Kya aapke paas portfolio hai haan {portfolio_link}")
            chunks.append(f"Aapne kaun se projects complete kiye hain {portfolio_link}")
            chunks.append(f"What projects have you completed {portfolio_link}")
            metadata.extend([{'type': 'portfolio'}] * 6)
        
        # Contact information - structured chunks with more context
        contact = data.get('contact', {})
        if contact:
            email = contact.get('email', '')
            phone = contact.get('phone', '')
            address = contact.get('address', '')
            website = contact.get('website', '')
            
            # Combined contact chunk with more semantic context (Urdu/Hindi/English variations)
            contact_full = f"Email {email} Phone {phone} Address {address} Website {website}"
            chunks.append(f"Contact information {contact_full}")
            chunks.append(f"Contact details {contact_full}")
            chunks.append(f"How to contact {contact_full}")
            chunks.append(f"Contact details batao {contact_full}")
            chunks.append(f"Contact kya hai {contact_full}")
            # Removed direct format to avoid matching with non-contact questions
            metadata.extend([{'type': 'contact'}] * 5)
            
            # Office address (separate for location questions)
            if address:
                chunks.append(f"Office address {address}")
                chunks.append(f"Office location {address}")
                chunks.append(f"Where is your office {address}")
                chunks.append(f"Aapka office kahan hai {address}")
                chunks.append(f"Office kahan hai {address}")
                metadata.extend([{'type': 'address'}] * 5)
        
        # Services - comprehensive chunks with more context
        services_list = data.get('services', [])
        if services_list:
            # Combined services list with more semantic variations (Urdu/Hindi/English)
            service_names = [s.get('name', '') for s in services_list if s.get('name')]
            if service_names:
                services_str = ', '.join(service_names)
                chunks.append(f"Services offered {services_str}")
                chunks.append(f"What services do you provide {services_str}")
                chunks.append(f"Services we provide {services_str}")
                chunks.append(f"What services do you offer {services_str}")
                chunks.append(f"Aap kya services dete hain answer {services_str}")
                chunks.append(f"Services kya hain answer {services_str}")
                chunks.append(f"Kya services dete hain answer {services_str}")
                chunks.append(f"Aap kya services dete hain {services_str}")
                chunks.append(f"Kya services dete hain {services_str}")
                chunks.append(f"Services list {services_str}")
                # Make services chunks very specific to avoid matching support/other FAQs
                chunks.append(f"What services list do you offer {services_str}")
                chunks.append(f"Company services list provided {services_str}")
                chunks.append(f"All services we provide list {services_str}")
                chunks.append(f"Services offered by company {services_str}")
                # More specific Urdu/Hindi variations
                metadata.extend([{'type': 'services_list'}] * 13)
            
            # Individual service details with Urdu/Hindi variations
            for service in services_list:
                service_name = service.get('name', '')
                service_details = service.get('details', '')
                if service_name and service_details:
                    chunks.append(f"Service {service_name} {service_details}")
                    # Add Urdu/Hindi service question variations
                    if 'Web' in service_name or 'website' in service_name.lower():
                        chunks.append(f"Kya aap websites banate hain {service_details}")
                    if 'Mobile' in service_name or 'app' in service_name.lower():
                        chunks.append(f"Kya aap mobile apps banate hain {service_details}")
                    if 'AI' in service_name or 'Artificial Intelligence' in service_name:
                        chunks.append(f"Kya aap AI systems develop karte hain {service_details}")
                    if 'Cloud' in service_name:
                        chunks.append(f"Kya aap cloud services dete hain {service_details}")
                    if 'UI/UX' in service_name or 'Design' in service_name:
                        chunks.append(f"Kya aap UI/UX design karte hain {service_details}")
                    metadata.append({'type': 'service', 'service_name': service_name})
                
                # Sub-services
                if service.get('sub_services'):
                    sub_services = ', '.join(service.get('sub_services', [])[:10])
                    chunks.append(f"{service_name} includes {sub_services}")
                    metadata.append({'type': 'sub_services', 'service_name': service_name})
        
        # Company values/rules - with more semantic context (Urdu/Hindi/English)
        additional = data.get('additional_info', {})
        if additional.get('company_values'):
            values_str = ', '.join(additional.get('company_values', []))
            chunks.append(f"Company values {values_str}")
            chunks.append(f"Company rules and policies {values_str}")
            chunks.append(f"What are the company rules {values_str}")
            chunks.append(f"Company policies {values_str}")
            chunks.append(f"Company k rules kya hain {values_str}")
            chunks.append(f"Rules kya hain company ke {values_str}")
            metadata.extend([{'type': 'company_values'}] * 6)
        
        # Payment methods - comprehensive chunks
        price_policy = data.get('price_policy', {})
        if price_policy.get('payment_methods'):
            payment_methods = ', '.join(price_policy.get('payment_methods', []))
            chunks.append(f"Payment methods {payment_methods}")
            chunks.append(f"What payment methods do you accept {payment_methods}")
            chunks.append(f"How do I pay {payment_methods}")
            chunks.append(f"Payment methods accept hain {payment_methods}")
            chunks.append(f"Kaun se payment methods accept hain {payment_methods}")
            chunks.append(f"Payment kaise karni hai {payment_methods}")
            metadata.extend([{'type': 'payment_methods'}] * 6)
        
        # Project process - comprehensive chunks (make very specific to avoid matching address)
        project_process = data.get('project_process', {})
        if project_process.get('english'):
            process_en = ' '.join(project_process.get('english', []))
            chunks.append(f"Project process development workflow steps {process_en}")
            chunks.append(f"How does your process work development {process_en}")
            chunks.append(f"Development process workflow steps {process_en}")
            chunks.append(f"Aapka development process kya hai steps {process_en}")
            chunks.append(f"Development kaise hoti hai process {process_en}")
            chunks.append(f"Aap projects kaise handle karte hain development process {process_en}")
            chunks.append(f"Development methodology process workflow {process_en}")
            chunks.append(f"What is your development methodology process {process_en}")
            # Add more specific process chunks to avoid address matching
            chunks.append(f"Aapka process kya hai development workflow {process_en}")
            chunks.append(f"Process kya hai development steps {process_en}")
            chunks.append(f"How do you handle projects development process {process_en}")
            chunks.append(f"Project handling process development workflow {process_en}")
            metadata.extend([{'type': 'project_process'}] * 12)
        
        if project_process.get('urdu'):
            process_ur = ' '.join(project_process.get('urdu', []))
            chunks.append(f"Project process Urdu {process_ur}")
            chunks.append(f"Development process Urdu {process_ur}")
            metadata.extend([{'type': 'project_process'}] * 2)
        
        # FAQ answers - make them more specific with question-like prefixes
        faq_answers = data.get('faq_answers', {})
        for key, answer in faq_answers.items():
            if answer:
                # Create multiple variations for better semantic matching
                key_clean = key.replace('_', ' ')
                chunks.append(f"{key_clean} {answer}")
                # Add question-like variations
                if key == 'nda':
                    chunks.append(f"Do you sign NDA {answer}")
                    chunks.append(f"Kya aap NDA sign karte hain {answer}")
                elif key == 'source_code':
                    chunks.append(f"Will I get source code {answer}")
                    chunks.append(f"Kya mujhe source code milega {answer}")
                    chunks.append(f"Do you give documentation {answer}")
                    chunks.append(f"Kya aap documentation dete hain {answer}")
                    chunks.append(f"Kya aap documentation provide karte hain {answer}")
                    chunks.append(f"Documentation provide karte hain {answer}")
                    chunks.append(f"Do you provide documentation {answer}")
                elif key == 'timeline':
                    chunks.append(f"Project timeline delivery time {answer}")
                    chunks.append(f"Project kitne time mein complete hoga {answer}")
                    chunks.append(f"Delivery time kya hai {answer}")
                    chunks.append(f"Typical timeline kya hai {answer}")
                elif key == 'technologies':
                    chunks.append(f"Which technologies do you use {answer}")
                    chunks.append(f"Aap kaun se technologies use karte hain {answer}")
                    chunks.append(f"Aap kaun se programming languages use karte hain {answer}")
                elif key == 'hosting':
                    chunks.append(f"Do you provide hosting {answer}")
                    chunks.append(f"Kya aap hosting provide karte hain {answer}")
                elif key == 'maintenance':
                    chunks.append(f"Do you provide maintenance {answer}")
                    chunks.append(f"Kya aap maintenance provide karte hain {answer}")
                elif key == 'support':
                    chunks.append(f"Do you provide customer support {answer}")
                    chunks.append(f"Kya aap customer support dete hain {answer}")
                    chunks.append(f"Support policy kya hai {answer}")
                elif key == 'revisions':
                    chunks.append(f"How many revisions do you offer {answer}")
                    chunks.append(f"Kitne revisions milte hain {answer}")
                elif key == 'payment_terms':
                    chunks.append(f"Do you take advance payment {answer}")
                    chunks.append(f"Kya aap advance payment lete hain {answer}")
                elif key == 'invoice':
                    chunks.append(f"Do you provide invoice {answer}")
                    chunks.append(f"Do you give invoice {answer}")
                    chunks.append(f"Kya aap invoice dete hain {answer}")
                    chunks.append(f"Kya aap invoice provide karte hain {answer}")
                    chunks.append(f"Invoice provide karte hain {answer}")
                    chunks.append(f"Do you send invoice {answer}")
                elif key == 'international':
                    chunks.append(f"Do you accept international payments {answer}")
                    chunks.append(f"Kya aap international payments accept karte hain {answer}")
                    chunks.append(f"international {answer}")
                elif key == 'pricing':
                    chunks.append(f"What are your prices {answer}")
                    chunks.append(f"How much does it cost {answer}")
                    chunks.append(f"What is your pricing model {answer}")
                    chunks.append(f"Kitna paisa lagega {answer}")
                    chunks.append(f"Price kya hai {answer}")
                    chunks.append(f"Pricing model kya hai {answer}")
                elif key == 'team_experience':
                    chunks.append(f"How experienced is your team {answer}")
                    chunks.append(f"Aapki team kitni experienced hai {answer}")
                    chunks.append(f"Team structure kya hai {answer}")
                    chunks.append(f"What is your team structure {answer}")
                elif key == 'communication':
                    chunks.append(f"How can I contact you {answer}")
                    chunks.append(f"Aap se kaise contact kar sakte hain {answer}")
                    chunks.append(f"Kya aapke paas WhatsApp hai {answer}")
                    chunks.append(f"Kya hum call par baat kar sakte hain {answer}")
                # Note: Generic chunk already added above, no need to add again
                metadata.append({'type': 'faq', 'topic': key})
        
        # Testimonials - remove company name to avoid semantic overlap with info questions
        for testimonial in data.get('testimonials', []):
            comment = testimonial.get('comment', '')
            if comment:
                # Remove company name from testimonials to prevent semantic matching with company info questions
                # Replace with generic terms to maintain semantic distance
                comment_clean = comment.replace(company_name, "the company").replace("Softerio Solutions", "the company")
                # Use semantically distinct prefix that won't match info questions
                chunks.append(f"Client review feedback testimonial customer experience story {comment_clean}")
                metadata.append({'type': 'testimonial'})
        
        self.dataset_chunks = chunks
        self.chunk_metadata = metadata
        print(f"✓ Loaded {len(chunks)} information chunks from dataset")
    
    def _find_relevant_chunks(self, question: str, top_k: int = 3, debug: bool = False) -> List[Tuple[str, float]]:
        """Find relevant chunks using pure semantic search"""
        
        if self.embedding_model is None or self.chunk_embeddings is None:
            return []
        
        try:
            # Encode question
            question_embedding = self.embedding_model.encode(
                [question],
                convert_to_numpy=True
            )
            
            # Calculate similarities
            similarities = cosine_similarity(question_embedding, self.chunk_embeddings)[0]
            
            # Get top matches
            top_indices = np.argsort(similarities)[::-1][:top_k * 5]
            
            # Debug output
            if debug:
                print(f"\n🔍 DEBUG for question: '{question}'")
                print("Top matches:")
                for i, idx in enumerate(top_indices[:10]):
                    chunk = self.dataset_chunks[idx]
                    sim = similarities[idx]
                    meta = self.chunk_metadata[idx] if idx < len(self.chunk_metadata) else {}
                    print(f"  {i+1}. [{sim:.3f}] [{meta.get('type', 'unknown')}] {chunk[:70]}...")
            
            # Filter by similarity threshold - pure semantic, no metadata filtering
            relevant = []
            for idx in top_indices:
                similarity = similarities[idx]
                chunk = self.dataset_chunks[idx]
                meta = self.chunk_metadata[idx] if idx < len(self.chunk_metadata) else {}
                
                # Pure semantic threshold - no hardcoding, no metadata filtering
                if similarity > 0.30:
                    relevant.append((chunk, similarity, meta))
            
            # Sort by similarity
            relevant.sort(reverse=True, key=lambda x: x[1])
            
            # Return top chunks with their metadata
            return [(chunk, sim) for chunk, sim, _ in relevant[:top_k]]
            
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []
    
    def _extract_answer_from_chunk(self, chunk: str) -> str:
        """Extract clean answer from chunk - remove question-like prefixes"""
        # Remove question-like prefixes that are part of chunk structure
        # These are metadata, not the actual answer content
        
        # Common patterns to remove (these are chunk metadata, not answers)
        patterns_to_remove = [
            "Client review feedback testimonial customer experience story ",
            "Contact details batao ",
            "Contact kya hai ",
            "Company name kya hai answer ",
            "Company ka naam hai ",
            "Company ka naam kya hai answer ",
            "Naam kya hai company ka answer ",
            "What is the company name answer ",
            "Company name is ",
            "Name of the company is ",
            "Aap kya services dete hain answer ",
            "Services kya hain answer ",
            "Kya services dete hain ",
            "What is the company name ",
            "Company name information ",
            "The company name is ",
            "Name of the company ",
            "Company name ",
            "Company description ",
            "About the company ",
            "Who we are ",
            "About us ",
            "Who are you ",
            "Tum kon ho ",
            "Aap kon hain ",
            "Contact information ",
            "Contact details ",
            "How to contact ",
            "What services do you provide ",
            "Services we provide ",
            "Services offered ",
            "Company k rules kya hain ",
            "Rules kya hain company ke ",
            "What are the company rules ",
            "Company rules and policies ",
            "Company policies ",
            "Company values ",
            "Payment methods ",
            "Service ",
            "Tell me about your company description ",
            "Aapki company kya karti hai description ",
            "What does your company do description ",
            "Company overview description ",
            "Company founded in ",
            "Founded year ",
            "When was the company founded ",
            "Aapki company kab bani ",
            "Company kab bani ",
            "Team size ",
            "Number of team members ",
            "How big is your team ",
            "Aapke team mein kitne log hain ",
            "Team mein kitne log hain ",
            "Office address ",
            "Office location ",
            "Where is your office ",
            "Aapka office kahan hai ",
            "Office kahan hai ",
            "Yes we work with clients from ",
            "International projects ",
            "Do you work internationally yes ",
            "Aap international projects karte hain haan ",
            "International projects haan ",
            "Portfolio ",
            "Do you have a portfolio yes ",
            "Portfolio website ",
            "Kya aapke paas portfolio hai haan ",
            "Aapne kaun se projects complete kiye hain ",
            "What projects have you completed ",
            "Kya aap websites banate hain ",
            "Kya aap mobile apps banate hain ",
            "Kya aap AI systems develop karte hain ",
            "Kya aap cloud services dete hain ",
            "Kya aap UI/UX design karte hain ",
            "What are your prices ",
            "How much does it cost ",
            "What is your pricing model ",
            "Kitna paisa lagega ",
            "Price kya hai ",
            "Pricing model kya hai ",
            "How experienced is your team ",
            "Aapki team kitni experienced hai ",
            "Team structure kya hai ",
            "What is your team structure ",
            "How can I contact you ",
            "Aap se kaise contact kar sakte hain ",
            "Kya aapke paas WhatsApp hai ",
            "Kya hum call par baat kar sakte hain ",
            "What services do you offer ",
            "Which technologies do you use ",
            "Do you provide customer support ",
            "Development kaise hoti hai ",
            "Project kitne time mein complete hoga ",
            "Delivery time kya hai ",
            "Typical timeline kya hai ",
            "Aap kaun se technologies use karte hain ",
            "Kya aap hosting provide karte hain ",
            "Project process development workflow steps ",
            "How does your process work development ",
            "Development process workflow steps ",
            "Aapka development process kya hai steps ",
            "Development kaise hoti hai process ",
            "Aap projects kaise handle karte hain development process ",
            "Development methodology process workflow ",
            "What is your development methodology process ",
            "Project process Urdu ",
            "Development process Urdu ",
            "Will I get source code ",
            "Kya mujhe source code milega ",
            "Do you give documentation ",
            "Kya aap documentation dete hain ",
            "Project timeline delivery time ",
            "Do you provide hosting ",
            "Do you provide maintenance ",
            "Kya aap maintenance provide karte hain ",
            "How many revisions do you offer ",
            "Kitne revisions milte hain ",
            "Do you take advance payment ",
            "Kya aap advance payment lete hain ",
            "Do you provide invoice ",
            "Do you give invoice ",
            "Kya aap invoice dete hain ",
            "Do you sign NDA ",
            "Kya aap NDA sign karte hain ",
            "Support policy kya hai ",
            "international ",
            "Do you accept international payments ",
            "Kya aap international payments accept karte hain ",
            "Kaun se payment methods accept hain ",
            "Payment kaise karni hai ",
            "What payment methods do you accept ",
            "How do I pay ",
            "Payment methods accept hain ",
            "Kya aapke paas packages hain ",
            "Kya aap jaldi delivery kar sakte hain ",
            "Kya aapke paas dedicated developers hain ",
            "Kya main monthly developers hire kar sakta hoon ",
            "Kya aap email support dete hain ",
            "Kya aap business automation karte hain ",
            "Do you build websites ",
            "Do you create mobile apps ",
            "Do you develop AI systems ",
            "Can you automate my business ",
            "Do you provide cloud services ",
            "Do you do UI/UX design ",
            "Can you develop blockchain solutions ",
            "Tell me about your AI services ",
            "Services list ",
            "Kya services dete hain ",
            "What services list do you offer ",
            "Company services list provided ",
            "All services we provide list ",
            "Services offered by company ",
            "Aapka process kya hai development workflow ",
            "Process kya hai development steps ",
            "How do you handle projects development process ",
            "Project handling process development workflow ",
            "About company information description ",
            "Company details description overview ",
            "Company information description ",
            "Tell me company description ",
            "Kya aap invoice provide karte hain ",
            "Invoice provide karte hain ",
            "Do you send invoice ",
            "Kya aap documentation provide karte hain ",
            "Documentation provide karte hain ",
            "Do you provide documentation ",
        ]
        
        answer = chunk
        # Remove longest matching prefix first (most specific)
        for pattern in sorted(patterns_to_remove, key=len, reverse=True):
            if answer.startswith(pattern):
                answer = answer[len(pattern):].strip()
                break
        
        return answer if answer else chunk
    
    def chat(self, user_input: str, language: str = "auto", debug: bool = False) -> str:
        """Main chat function - Pure semantic search, model learns from dataset"""
        user_input = user_input.strip()
        if not user_input:
            return "Please ask me something about Softerio Solutions!"
        
        # Find relevant chunks using semantic search (model learns from dataset)
        relevant_chunks = self._find_relevant_chunks(user_input, top_k=3, debug=debug)
        
        if not relevant_chunks:
            return "I understand your question. Let me help you with information about Softerio Solutions. Could you be more specific?"
        
        # Get best matching chunk (model's learned answer from dataset)
        best_chunk, similarity = relevant_chunks[0]
        
        # Extract clean answer (minimal cleaning, no hardcoding)
        answer = self._extract_answer_from_chunk(best_chunk)
        
        return answer


def main():
    """Main function to run the chatbot"""
    print("=" * 60)
    print("🤖 Softerio Solutions Chatbot")
    print("=" * 60)
    print("\nType 'quit' or 'exit' to stop\n")
    
    chatbot = CompanyChatbot()
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nThank you for chatting with Softerio Solutions! 👋")
                break
            
            response = chatbot.chat(user_input)
            print(f"\nBot: {response}")
            
        except KeyboardInterrupt:
            print("\n\nThank you for chatting! 👋")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
