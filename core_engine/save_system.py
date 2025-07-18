"""
🌌 INFINITUS Save System
Advanced save/load system with compression, versioning, and data integrity.
"""

import asyncio
import json
import os
import time
import hashlib
import zlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import threading
import shutil

try:
    import msgpack
    import lz4.frame
    COMPRESSION_AVAILABLE = True
except ImportError:
    COMPRESSION_AVAILABLE = False

class SaveFormat(Enum):
    """Save file formats."""
    JSON = "json"
    MSGPACK = "msgpack"
    BINARY = "binary"

class CompressionType(Enum):
    """Compression algorithms."""
    NONE = "none"
    ZLIB = "zlib"
    LZ4 = "lz4"

@dataclass
class SaveMetadata:
    """Save file metadata."""
    version: str = "1.0.0"
    timestamp: float = 0.0
    game_version: str = "1.0.0-alpha"
    world_name: str = ""
    player_name: str = ""
    playtime: float = 0.0
    save_count: int = 0
    checksum: str = ""
    format: SaveFormat = SaveFormat.JSON
    compression: CompressionType = CompressionType.ZLIB
    size_uncompressed: int = 0
    size_compressed: int = 0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

@dataclass
class SaveSlot:
    """Save slot information."""
    slot_id: str
    name: str
    metadata: SaveMetadata
    path: str
    exists: bool = False
    corrupted: bool = False
    
class SaveSystem:
    """Advanced save/load system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Save configuration
        self.save_dir = Path("saves")
        self.backup_dir = Path("saves/backups")
        self.temp_dir = Path("saves/temp")
        
        # Save settings
        self.max_saves_per_world = 10
        self.max_backups = 5
        self.auto_backup_interval = 300.0  # 5 minutes
        self.compression_enabled = True
        self.compression_type = CompressionType.LZ4 if COMPRESSION_AVAILABLE else CompressionType.ZLIB
        self.save_format = SaveFormat.MSGPACK if COMPRESSION_AVAILABLE else SaveFormat.JSON
        
        # Save slots
        self.save_slots: Dict[str, SaveSlot] = {}
        self.current_save_slot: Optional[str] = None
        
        # Threading
        self.save_lock = threading.RLock()
        self.background_save_enabled = True
        
        # Statistics
        self.save_count = 0
        self.load_count = 0
        self.total_save_time = 0.0
        self.total_load_time = 0.0
        
        # Auto-save
        self.auto_save_enabled = True
        self.last_auto_save = 0.0
        
        self.logger.info("💾 Save System initialized")
    
    async def initialize(self):
        """Initialize the save system."""
        self.logger.info("🔧 Initializing Save System...")
        
        # Create directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Scan for existing saves
        await self._scan_saves()
        
        # Start background tasks
        if self.background_save_enabled:
            asyncio.create_task(self._background_save_loop())
        
        self.logger.info("✅ Save System initialization complete")
    
    async def _scan_saves(self):
        """Scan for existing save files."""
        self.logger.info("🔍 Scanning for save files...")
        
        save_count = 0
        for save_file in self.save_dir.glob("*.save"):
            try:
                metadata = await self._load_metadata(save_file)
                if metadata:
                    slot_id = save_file.stem
                    save_slot = SaveSlot(
                        slot_id=slot_id,
                        name=metadata.world_name or slot_id,
                        metadata=metadata,
                        path=str(save_file),
                        exists=True
                    )
                    
                    # Verify save integrity
                    if await self._verify_save_integrity(save_file):
                        self.save_slots[slot_id] = save_slot
                        save_count += 1
                    else:
                        save_slot.corrupted = True
                        self.save_slots[slot_id] = save_slot
                        self.logger.warning(f"Corrupted save file: {save_file}")
                        
            except Exception as e:
                self.logger.error(f"Error scanning save file {save_file}: {e}")
        
        self.logger.info(f"✅ Found {save_count} valid save files")
    
    async def save_world(self, world_name: str, world_data: Dict[str, Any], 
                        slot_id: Optional[str] = None) -> bool:
        """Save world data to a save file."""
        try:
            start_time = time.time()
            
            # Generate slot ID if not provided
            if not slot_id:
                slot_id = f"{world_name}_{int(time.time())}"
            
            # Create save data structure
            save_data = {
                'world_data': world_data,
                'metadata': {
                    'world_name': world_name,
                    'timestamp': time.time(),
                    'game_version': '1.0.0-alpha',
                    'save_count': self.save_count + 1
                }
            }
            
            # Save to file
            save_path = self.save_dir / f"{slot_id}.save"
            success = await self._save_to_file(save_data, save_path)
            
            if success:
                # Update save slot
                metadata = SaveMetadata(
                    world_name=world_name,
                    timestamp=time.time(),
                    save_count=self.save_count + 1
                )
                
                save_slot = SaveSlot(
                    slot_id=slot_id,
                    name=world_name,
                    metadata=metadata,
                    path=str(save_path),
                    exists=True
                )
                
                self.save_slots[slot_id] = save_slot
                self.current_save_slot = slot_id
                
                # Update statistics
                self.save_count += 1
                self.total_save_time += time.time() - start_time
                
                # Create backup
                await self._create_backup(save_path)
                
                self.logger.info(f"✅ World saved: {world_name} ({time.time() - start_time:.3f}s)")
                return True
            else:
                self.logger.error(f"❌ Failed to save world: {world_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error saving world {world_name}: {e}")
            return False
    
    async def load_world(self, world_name: str) -> Optional[Dict[str, Any]]:
        """Load world data from a save file."""
        try:
            start_time = time.time()
            
            # Find save slot
            save_slot = None
            for slot in self.save_slots.values():
                if slot.name == world_name or slot.slot_id == world_name:
                    save_slot = slot
                    break
            
            if not save_slot:
                self.logger.warning(f"Save not found: {world_name}")
                return None
            
            if save_slot.corrupted:
                self.logger.error(f"Save file corrupted: {world_name}")
                return None
            
            # Load from file
            save_data = await self._load_from_file(Path(save_slot.path))
            
            if save_data and 'world_data' in save_data:
                # Update statistics
                self.load_count += 1
                self.total_load_time += time.time() - start_time
                
                self.current_save_slot = save_slot.slot_id
                self.logger.info(f"✅ World loaded: {world_name} ({time.time() - start_time:.3f}s)")
                return save_data['world_data']
            else:
                self.logger.error(f"❌ Invalid save data: {world_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error loading world {world_name}: {e}")
            return None
    
    async def _save_to_file(self, data: Dict[str, Any], file_path: Path) -> bool:
        """Save data to a file with compression and error handling."""
        try:
            with self.save_lock:
                # Serialize data
                if self.save_format == SaveFormat.MSGPACK and COMPRESSION_AVAILABLE:
                    serialized_data = msgpack.packb(data)
                else:
                    serialized_data = json.dumps(data, indent=2).encode('utf-8')
                
                # Compress data
                if self.compression_enabled:
                    if self.compression_type == CompressionType.LZ4 and COMPRESSION_AVAILABLE:
                        compressed_data = lz4.frame.compress(serialized_data)
                    elif self.compression_type == CompressionType.ZLIB:
                        compressed_data = zlib.compress(serialized_data)
                    else:
                        compressed_data = serialized_data
                else:
                    compressed_data = serialized_data
                
                # Calculate checksum
                checksum = hashlib.sha256(compressed_data).hexdigest()
                
                # Create metadata
                metadata = SaveMetadata(
                    timestamp=time.time(),
                    checksum=checksum,
                    format=self.save_format,
                    compression=self.compression_type,
                    size_uncompressed=len(serialized_data),
                    size_compressed=len(compressed_data)
                )
                
                # Create final save structure
                save_structure = {
                    'metadata': asdict(metadata),
                    'data': compressed_data
                }
                
                # Write to temporary file first
                temp_path = self.temp_dir / f"{file_path.name}.tmp"
                
                if self.save_format == SaveFormat.MSGPACK and COMPRESSION_AVAILABLE:
                    with open(temp_path, 'wb') as f:
                        msgpack.pack(save_structure, f)
                else:
                    # For JSON, we need to encode binary data
                    save_structure['data'] = compressed_data.hex()
                    with open(temp_path, 'w', encoding='utf-8') as f:
                        json.dump(save_structure, f, indent=2)
                
                # Atomic move to final location
                shutil.move(str(temp_path), str(file_path))
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error saving to file {file_path}: {e}")
            return False
    
    async def _load_from_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load data from a file with decompression and error handling."""
        try:
            with self.save_lock:
                # Load save structure
                if file_path.suffix == '.save':
                    # Try to detect format
                    with open(file_path, 'rb') as f:
                        header = f.read(1)
                        f.seek(0)
                        
                        if header == b'{':
                            # JSON format
                            content = f.read().decode('utf-8')
                            save_structure = json.loads(content)
                            
                            # Decode binary data from hex
                            if isinstance(save_structure.get('data'), str):
                                save_structure['data'] = bytes.fromhex(save_structure['data'])
                        else:
                            # MessagePack format
                            if COMPRESSION_AVAILABLE:
                                save_structure = msgpack.unpack(f)
                            else:
                                self.logger.error("MessagePack not available")
                                return None
                else:
                    # Legacy format
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                
                # Extract metadata and data
                metadata = save_structure.get('metadata', {})
                compressed_data = save_structure.get('data', b'')
                
                # Verify checksum
                if metadata.get('checksum'):
                    calculated_checksum = hashlib.sha256(compressed_data).hexdigest()
                    if calculated_checksum != metadata['checksum']:
                        self.logger.error(f"Checksum mismatch for {file_path}")
                        return None
                
                # Decompress data
                compression_type = CompressionType(metadata.get('compression', 'none'))
                
                if compression_type == CompressionType.LZ4 and COMPRESSION_AVAILABLE:
                    decompressed_data = lz4.frame.decompress(compressed_data)
                elif compression_type == CompressionType.ZLIB:
                    decompressed_data = zlib.decompress(compressed_data)
                else:
                    decompressed_data = compressed_data
                
                # Deserialize data
                save_format = SaveFormat(metadata.get('format', 'json'))
                
                if save_format == SaveFormat.MSGPACK and COMPRESSION_AVAILABLE:
                    data = msgpack.unpackb(decompressed_data, raw=False)
                else:
                    data = json.loads(decompressed_data.decode('utf-8'))
                
                return data
                
        except Exception as e:
            self.logger.error(f"Error loading from file {file_path}: {e}")
            return None
    
    async def _load_metadata(self, file_path: Path) -> Optional[SaveMetadata]:
        """Load only metadata from a save file."""
        try:
            # Quick metadata extraction without loading full file
            with open(file_path, 'rb') as f:
                header = f.read(1)
                f.seek(0)
                
                if header == b'{':
                    # JSON format - read until we find metadata
                    content = f.read(1024).decode('utf-8')  # Read first 1KB
                    if '"metadata"' in content:
                        # Load full file for now (could be optimized)
                        f.seek(0)
                        save_structure = json.load(f)
                        metadata_dict = save_structure.get('metadata', {})
                        return SaveMetadata(**metadata_dict)
                else:
                    # MessagePack format
                    if COMPRESSION_AVAILABLE:
                        save_structure = msgpack.unpack(f)
                        metadata_dict = save_structure.get('metadata', {})
                        return SaveMetadata(**metadata_dict)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading metadata from {file_path}: {e}")
            return None
    
    async def _verify_save_integrity(self, file_path: Path) -> bool:
        """Verify the integrity of a save file."""
        try:
            # Try to load the file
            data = await self._load_from_file(file_path)
            return data is not None
            
        except Exception:
            return False
    
    async def _create_backup(self, save_path: Path):
        """Create a backup of a save file."""
        try:
            timestamp = int(time.time())
            backup_name = f"{save_path.stem}_{timestamp}.backup"
            backup_path = self.backup_dir / backup_name
            
            shutil.copy2(str(save_path), str(backup_path))
            
            # Clean up old backups
            await self._cleanup_old_backups(save_path.stem)
            
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
    
    async def _cleanup_old_backups(self, save_name: str):
        """Clean up old backup files."""
        try:
            backups = list(self.backup_dir.glob(f"{save_name}_*.backup"))
            backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Keep only the most recent backups
            for backup in backups[self.max_backups:]:
                backup.unlink()
                
        except Exception as e:
            self.logger.error(f"Error cleaning up backups: {e}")
    
    async def _background_save_loop(self):
        """Background save loop for auto-save functionality."""
        while self.background_save_enabled:
            try:
                # Check if auto-save is needed
                current_time = time.time()
                if (self.auto_save_enabled and 
                    current_time - self.last_auto_save >= self.auto_backup_interval):
                    
                    # This would trigger auto-save in the game engine
                    # For now, just update the timestamp
                    self.last_auto_save = current_time
                
                await asyncio.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in background save loop: {e}")
                await asyncio.sleep(30.0)  # Wait longer on error
    
    def get_save_slots(self) -> List[SaveSlot]:
        """Get all available save slots."""
        return list(self.save_slots.values())
    
    def get_save_slot(self, slot_id: str) -> Optional[SaveSlot]:
        """Get a specific save slot."""
        return self.save_slots.get(slot_id)
    
    def delete_save(self, slot_id: str) -> bool:
        """Delete a save file."""
        try:
            if slot_id in self.save_slots:
                save_slot = self.save_slots[slot_id]
                save_path = Path(save_slot.path)
                
                if save_path.exists():
                    save_path.unlink()
                
                del self.save_slots[slot_id]
                
                if self.current_save_slot == slot_id:
                    self.current_save_slot = None
                
                self.logger.info(f"Deleted save: {slot_id}")
                return True
            else:
                self.logger.warning(f"Save slot not found: {slot_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error deleting save {slot_id}: {e}")
            return False
    
    def export_save(self, slot_id: str, export_path: str) -> bool:
        """Export a save file to a different location."""
        try:
            if slot_id in self.save_slots:
                save_slot = self.save_slots[slot_id]
                source_path = Path(save_slot.path)
                target_path = Path(export_path)
                
                shutil.copy2(str(source_path), str(target_path))
                
                self.logger.info(f"Exported save {slot_id} to {export_path}")
                return True
            else:
                self.logger.warning(f"Save slot not found: {slot_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error exporting save {slot_id}: {e}")
            return False
    
    def import_save(self, import_path: str, slot_id: Optional[str] = None) -> bool:
        """Import a save file from a different location."""
        try:
            source_path = Path(import_path)
            
            if not source_path.exists():
                self.logger.error(f"Import file not found: {import_path}")
                return False
            
            # Generate slot ID if not provided
            if not slot_id:
                slot_id = f"imported_{int(time.time())}"
            
            target_path = self.save_dir / f"{slot_id}.save"
            shutil.copy2(str(source_path), str(target_path))
            
            # Scan the new save
            asyncio.create_task(self._scan_saves())
            
            self.logger.info(f"Imported save from {import_path} as {slot_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error importing save from {import_path}: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get save system statistics."""
        return {
            'total_saves': len(self.save_slots),
            'corrupted_saves': len([s for s in self.save_slots.values() if s.corrupted]),
            'save_count': self.save_count,
            'load_count': self.load_count,
            'average_save_time': self.total_save_time / self.save_count if self.save_count > 0 else 0,
            'average_load_time': self.total_load_time / self.load_count if self.load_count > 0 else 0,
            'compression_enabled': self.compression_enabled,
            'compression_type': self.compression_type.value,
            'save_format': self.save_format.value
        }
    
    async def shutdown(self):
        """Shutdown the save system."""
        self.logger.info("🔄 Shutting down Save System...")
        
        # Disable background saving
        self.background_save_enabled = False
        
        # Wait for any pending saves to complete
        await asyncio.sleep(0.1)
        
        # Clean up temporary files
        try:
            for temp_file in self.temp_dir.glob("*.tmp"):
                temp_file.unlink()
        except Exception as e:
            self.logger.error(f"Error cleaning up temp files: {e}")
        
        self.logger.info("✅ Save System shutdown complete")