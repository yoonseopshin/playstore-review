"""
File and directory utilities
"""
import os
import platform


class FileUtils:
    """File and directory management utilities"""
    
    @staticmethod
    def setup_environment(output_dir: str = "./output"):
        """Setup output environment and font settings"""
        
        print(f"Current working directory: {os.getcwd()}")
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"✓ Output directory created successfully: {os.path.abspath(output_dir)}")
        else:
            print(f"✓ Output directory exists: {os.path.abspath(output_dir)}")
        
        # Create .nojekyll for GitHub Pages
        nojekyll_file = os.path.join(output_dir, '.nojekyll')
        if not os.path.exists(nojekyll_file):
            with open(nojekyll_file, 'w') as f:
                f.write('')
            print(f"✓ .nojekyll file created: {nojekyll_file}")
        
        return os.path.abspath(output_dir)
    
    @staticmethod
    def get_font_settings():
        """Get font settings for current OS"""
        font_settings = {
            'darwin': {  # macOS
                'name': "AppleGothic",
                'path': "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
            },
            'linux': {
                'name': "NanumGothic", 
                'path': "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
            },
            'windows': {
                'name': "Malgun Gothic",
                'path': "C:/Windows/Fonts/malgun.ttf"
            }
        }
        
        current_os = platform.system().lower()
        settings = font_settings.get(current_os, font_settings['linux'])
        
        return settings['name'], settings['path']
